"""ColBERT: late interaction over per-token BERT embeddings.

This module implements a ColBERT-style dual scorer using the
:class:`~xpmir.text.encoders.TokensRepresentationOutput` abstraction. The query
and document encoders return one vector per input token; scoring is done using
the late-interaction "MaxSim" operator: for each query token we take the
maximum similarity against all document tokens, then sum over the query
tokens.

Reference: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search
via Contextualized Late Interaction over BERT" (SIGIR 2020).
"""

from pathlib import Path
from typing import List, Optional
from attrs import evolve

import torch
import torch.nn as nn
from experimaestro import Param, field
from datamaestro_ir.data import IDTextRecord
from xpm_torch.learner import TrainerContext

from xpmir.neural.dual import DualVectorScorer
from xpmir.text.encoders import (
    TokensRepresentationOutput,
    TokenizedTextEncoderBase,
)
from xpmir.text.tokenizers import TokenizerOptions


class ColBERTEncoder(
    DualVectorScorer[TokensRepresentationOutput, TokensRepresentationOutput]
):
    """ColBERT-style dual scorer with late interaction MaxSim.

    The document (and optional query) encoder must return
    :class:`~xpmir.text.encoders.TokensRepresentationOutput`, i.e. a
    ``(batch, max_tokens, hidden_dim)`` tensor together with the tokenized
    inputs (providing the attention mask). A trainable linear projection
    reduces the per-token vectors to ``dim`` and the vectors are L2-normalised
    so the dot product amounts to a cosine similarity.
    """

    encoder: Param[TokenizedTextEncoderBase[IDTextRecord, TokensRepresentationOutput]]
    """The document token encoder (returns one vector per token)."""

    query_encoder: Param[
        Optional[TokenizedTextEncoderBase[IDTextRecord, TokensRepresentationOutput]]
    ]
    """Optional separate query encoder. When unset, the document encoder is
    used for queries too."""

    dim: Param[int] = field(default=128, ignore_default=True)
    """Output dimension of the per-token projection."""

    query_maxlen: Param[int] = field(default=32, ignore_default=True)
    """Maximum number of tokens kept for a query."""

    doc_maxlen: Param[int] = field(default=180, ignore_default=True)
    """Maximum number of tokens kept for a document."""

    def __initialize__(self):
        super().__initialize__()
        hidden = self.encoder.dimension
        self._projection = nn.Linear(hidden, self.dim, bias=False)

    @property
    def dimension(self) -> int:
        """Projection dimension (returned per token)."""
        return self.dim

    # ------------------------------------------------------------------ utils

    def _project(
        self, output: TokensRepresentationOutput
    ) -> TokensRepresentationOutput:
        """Project the per-token vectors to ``dim`` and L2-normalise them."""
        value = self._projection(output.value)
        value = torch.nn.functional.normalize(value, p=2, dim=-1)
        return evolve(output, value=value)

    @staticmethod
    def _token_mask(output: TokensRepresentationOutput) -> Optional[torch.Tensor]:
        mask = output.tokenized.mask
        if mask is None:
            return None
        return mask.to(output.value.device).bool()

    def document_token_embeddings(
        self, records: List[IDTextRecord]
    ) -> List[torch.Tensor]:
        """Encode a batch of documents and return the list of per-token
        embeddings, one tensor ``(num_tokens, dim)`` per document. Padding
        positions are filtered out.
        """
        output = self.encode_documents(records)
        mask = self._token_mask(output)
        value = output.value
        if mask is None:
            return [value[i] for i in range(value.shape[0])]
        return [value[i][mask[i]] for i in range(value.shape[0])]

    def query_token_embeddings(self, records: List[IDTextRecord]) -> torch.Tensor:
        """Encode a batch of queries and return a dense
        ``(batch, query_maxlen, dim)`` tensor suitable for fast-plaid search.
        """
        return self.encode_queries(records).value

    # ------------------------------------------------------------- encoding

    def encode_queries(self, records: List[IDTextRecord]) -> TokensRepresentationOutput:
        options = TokenizerOptions(max_length=self.query_maxlen)
        output = self._query_encoder(records, options=options)
        return self._project(output)

    def encode_documents(
        self, records: List[IDTextRecord]
    ) -> TokensRepresentationOutput:
        options = TokenizerOptions(max_length=self.doc_maxlen)
        output = self.encoder(records, options=options)
        return self._project(output)

    # --------------------------------------------------------------- scoring

    def _max_sim(
        self,
        queries: TokensRepresentationOutput,
        documents: TokensRepresentationOutput,
        all_pairs: bool,
    ) -> torch.Tensor:
        """Compute the MaxSim operator.

        When ``all_pairs`` is True, returns an ``(Nq, Nd)`` matrix of scores
        between every query and every document; otherwise returns a vector of
        ``Nq == Nd`` scores for the aligned query/document pairs.
        """
        q = queries.value
        d = documents.value
        doc_mask = self._token_mask(documents)
        query_mask = self._token_mask(queries)
        neg_inf = torch.finfo(q.dtype).min

        if all_pairs:
            # scores: (Nq, Lq, Nd, Ld)
            scores = torch.einsum("qmd,nkd->qmnk", q, d)
            if doc_mask is not None:
                scores = scores.masked_fill(
                    ~doc_mask.unsqueeze(0).unsqueeze(0), neg_inf
                )
            # max over doc tokens -> (Nq, Lq, Nd)
            max_scores = scores.max(dim=-1).values
            if query_mask is not None:
                max_scores = max_scores * query_mask.unsqueeze(-1).to(max_scores.dtype)
            return max_scores.sum(dim=1)

        # Paired scoring: expects Nq == Nd
        scores = torch.einsum("nmd,nkd->nmk", q, d)
        if doc_mask is not None:
            scores = scores.masked_fill(~doc_mask.unsqueeze(1), neg_inf)
        max_scores = scores.max(dim=-1).values  # (N, Lq)
        if query_mask is not None:
            max_scores = max_scores * query_mask.to(max_scores.dtype)
        return max_scores.sum(dim=-1)

    def score_product(
        self,
        queries: TokensRepresentationOutput,
        documents: TokensRepresentationOutput,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        return self._max_sim(queries, documents, all_pairs=True)

    def score_pairs(
        self,
        queries: TokensRepresentationOutput,
        documents: TokensRepresentationOutput,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        return self._max_sim(queries, documents, all_pairs=False)

    # ------------------------------------------------------ (de)serialisation

    def save_model(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_model(path / "encoder")
        if self.query_encoder is not None and self.query_encoder is not self.encoder:
            self._query_encoder.save_model(path / "query_encoder")
        torch.save(self._projection.state_dict(), path / "projection.pth")

    def load_model(self, path: Path):
        if (path / "encoder").exists():
            self.encoder.load_model(path / "encoder")
            if (path / "query_encoder").exists() and self.query_encoder is not None:
                self._query_encoder.load_model(path / "query_encoder")
        proj_path = path / "projection.pth"
        if proj_path.exists():
            self._projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
