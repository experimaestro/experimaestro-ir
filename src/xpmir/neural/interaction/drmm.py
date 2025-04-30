import math
import numpy as np
from typing import Optional, List
from experimaestro import Config, Param, default
import torch
from torch import nn
from typing_extensions import Annotated
from xpmir.index import Index
import xpmir.neural.modules as modules
from xpmir.neural.interaction import (
    InteractionScorer,
    SimilarityOutput,
    TrainerContext,
    TokenizedTextEncoderBase,
    TokenizerOptions,
    TokensEncoderOutput,
)
from xpmir.text.encoders import InputType
from .common import SimilarityInputWithTokens

# The code below is heavily borrowed from OpenNIR


class CountHistogram(Config, nn.Module):
    """Base histogram class

    Attributes:
        nbins: number of bins in matching histogram
    """

    nbins: Param[int] = 29

    def forward(self, simmat: torch.Tensor, dlens: List[int], mask: torch.BoolTensor):
        """Computes the histograms for each query term

        :param simmat: A (B... x Lq x Ld) matrix
        :param mask: A (B... x Lq x Ld) mask
        :param dlens: The document lengths (vector of size D)
        :return: A (B... x Lq x nbins) matrix containing counts
        """
        # +1e-5 to nudge scores of 1 to above threshold
        bins = ((simmat + 1.00001) / 2.0 * (self.nbins - 1)).long()
        weights = mask.float()

        hist = torch.zeros(
            *simmat.shape[:-1], self.nbins, device=simmat.device, dtype=simmat.dtype
        )
        return hist.scatter_add_(simmat.ndim - 1, bins, weights)


class NormalizedHistogram(CountHistogram):
    def forward(self, simmat, dlens, mask):
        result = super().forward(simmat, dlens, mask)
        BATCH, QLEN, _ = simmat.shape
        return result / dlens.reshape(BATCH, 1).expand(BATCH, QLEN)


class LogCountHistogram(CountHistogram):
    def forward(self, simmat, dlens, mask):
        result = super().forward(simmat, dlens, mask)
        return (result.float() + 1e-5).log()


class Combination(Config, nn.Module):
    def forward(self, scores: torch.Tensor, idf: torch.Tensor):
        """Combines term scores with IDF

        :param scores: A (B... x Lq) tensor
        :param idf: A (B... x Lq) tensor
        """
        ...


class SumCombination(Combination):
    def forward(self, scores, idf):
        return scores.sum(dim=-1)


class IdfCombination(Combination):
    def forward(self, scores: torch.Tensor, idf: torch.Tensor):
        idf = idf.softmax(dim=-1)
        return (scores * idf).sum(dim=-1)


class Drmm(InteractionScorer):
    """Deep Relevance Matching Model (DRMM)

    Implementation of the DRMM model from:

      Jiafeng Guo, Yixing Fan, Qingyao Ai, and William Bruce Croft. 2016. A Deep
      Relevance Matching Model for Ad-hoc Retrieval. In CIKM.
    """

    hist: Annotated[CountHistogram, default(LogCountHistogram())]
    """The histogram type"""

    hidden: Param[int] = 5
    """Hidden layer dimension for the feed forward matching network"""

    index: Param[Optional[Index]]
    """The index (only used when using IDF to combine)"""

    combine: Annotated[Combination, default(IdfCombination())]
    """How to combine the query term scores"""

    def __validate__(self):
        super().__validate__()
        assert (self.combine != "idf") or (
            self.index is not None
        ), "index must be provided if using IDF"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.simmat = modules.InteractionMatrix(self.encoder.pad_tokenid)
        self.hidden_1 = nn.Linear(self.hist.nbins, self.hidden)
        self.hidden_2 = nn.Linear(self.hidden, 1)
        self.needs_idf = isinstance(self.combine, IdfCombination)

    def _encode(
        self,
        texts: List[InputType],
        encoder: TokenizedTextEncoderBase[InputType, TokensEncoderOutput],
        options: TokenizerOptions,
    ) -> SimilarityInputWithTokens:
        encoded = encoder(texts, options=options)

        max_len = max(encoded.tokenized.lens)
        padded_tokens = [
            (t + [""] * (max_len - len(t))) for t in encoded.tokenized.tokens
        ]

        return self.similarity.preprocess(
            SimilarityInputWithTokens(
                encoded.value,
                encoded.tokenized.mask,
                np.array(padded_tokens, dtype=str),
            )
        )

    def compute_scores(
        self,
        queries: SimilarityInputWithTokens,
        documents: SimilarityInputWithTokens,
        value: SimilarityOutput,
        info: Optional[TrainerContext] = None,
    ):
        """Compute the scores given the tensor of similarities (B x Lq x Ld) or
        (Bq x Lq x Bd x Ld)"""
        # Computes the IDF if needed
        query_idf = None
        if self.needs_idf:
            assert self.index is not None
            query_idf = torch.full_like(queries.mask, float("-inf"), dtype=torch.float)
            log_nd = math.log(self.index.documentcount + 1)
            for i, tokens_i in enumerate(queries.tokens):
                for j, t in enumerate(tokens_i):
                    query_idf[i, j] = log_nd - math.log(self.index.term_df(t) + 1)

        mask = value.q_view(queries.mask) * value.d_view(documents.mask)
        similarity = value.similarity
        dlens = torch.LongTensor([len(tokens) for tokens in documents.tokens])

        if value.similarity.ndim == 4:
            # Transform into B... x Lq x Ld shape
            similarity = value.similarity.transpose(1, 2)
            mask = mask.transpose(1, 2)
            query_idf = query_idf.unsqueeze(0)

        qterm_features = self.histogram_pool(similarity, dlens, mask)
        qterm_scores = self.hidden_2(torch.relu(self.hidden_1(qterm_features))).squeeze(
            -1
        )
        return self.combine(qterm_scores, query_idf)

    def histogram_pool(self, simmat, dlens, mask):
        histogram = self.hist(simmat, dlens, mask)
        return histogram
