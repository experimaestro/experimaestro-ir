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
from xpmir.text.encoders import TokensRepresentationOutput
from xpmir.text.tokenizers import TokenizedTexts, TokenizerOptions
from xpm_torch.utils import to_device


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

    The :attr:`encoder` (and :attr:`query_encoder`) inherited from
    :class:`~xpmir.neural.dual.DualVectorScorer` must in practice be a
    :class:`~xpmir.text.encoders.TokenizedTextEncoder` returning
    :class:`~xpmir.text.encoders.TokensRepresentationOutput`. The
    :class:`~xpmir.text.encoders.TokenizedTextEncoder` exposes the
    ``tokenize`` / ``forward_tokenized`` split that query augmentation needs.
    """

    dim: Param[int] = field(default=128, ignore_default=True)
    """Output dimension of the per-token projection."""

    query_maxlen: Param[int] = field(default=32, ignore_default=True)
    """Maximum number of tokens kept for a query."""

    doc_maxlen: Param[int] = field(default=180, ignore_default=True)
    """Maximum number of tokens kept for a document."""

    query_augmentation: Param[bool] = field(default=True)
    """Whether to apply ColBERT's query augmentation: queries shorter than
    ``query_maxlen`` are right-padded with ``[MASK]`` tokens (instead of
    ``[PAD]``) and every position participates in MaxSim. This mirrors the
    original ColBERT implementation. Disable to use plain padded queries with
    padding excluded from MaxSim."""

    def __initialize__(self):
        super().__initialize__()
        hidden = self.encoder.dimension
        self._projection = nn.Linear(hidden, self.dim, bias=False)
        if self.query_augmentation:
            self._mask_token_id = self._lookup_mask_token_id(self._query_encoder)

    @staticmethod
    def _lookup_mask_token_id(encoder) -> int:
        """Locate the underlying HF tokenizer's ``mask_token_id`` by walking
        nested ``tokenizer`` attributes."""
        obj = encoder
        seen = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            mask_id = getattr(obj, "mask_token_id", None)
            if mask_id is not None:
                return mask_id
            obj = getattr(obj, "tokenizer", None)
        raise ValueError(
            "Could not locate mask_token_id on the query encoder; set "
            "query_augmentation=False or use an encoder backed by an HF "
            "tokenizer exposing [MASK]."
        )

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
        return to_device(mask, output.value.device).bool()

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
        if self.query_augmentation:
            tokenized = self._query_encoder.tokenize(records, options=options)
            tokenized = self._mask_pad_query(tokenized)
            output = self._query_encoder.forward_tokenized(tokenized)
        else:
            output = self._query_encoder(records, options=options)
        return self._project(output)

    def _mask_pad_query(self, tokenized: TokenizedTexts) -> TokenizedTexts:
        """Apply ColBERT query augmentation: right-pad to ``query_maxlen``,
        replace every padding position with ``[MASK]`` and set the attention
        mask to 1 over those positions so they participate in MaxSim.
        """
        ids = tokenized.ids
        mask = tokenized.mask
        token_type_ids = tokenized.token_type_ids
        batch_size, current_len = ids.shape

        pad_len = self.query_maxlen - current_len
        if pad_len > 0:
            id_pad = torch.zeros(
                (batch_size, pad_len), dtype=ids.dtype, device=ids.device
            )
            ids = torch.cat([ids, id_pad], dim=1)
            if mask is not None:
                mask_pad = torch.zeros(
                    (batch_size, pad_len), dtype=mask.dtype, device=mask.device
                )
                mask = torch.cat([mask, mask_pad], dim=1)
            if token_type_ids is not None:
                tt_pad = torch.zeros(
                    (batch_size, pad_len),
                    dtype=token_type_ids.dtype,
                    device=token_type_ids.device,
                )
                token_type_ids = torch.cat([token_type_ids, tt_pad], dim=1)

        if mask is not None:
            pad_positions = mask == 0
            ids = ids.masked_fill(pad_positions, self._mask_token_id)
            mask = torch.ones_like(mask)

        return TokenizedTexts(
            tokens=tokenized.tokens,
            ids=ids,
            lens=[self.query_maxlen] * batch_size,
            mask=mask,
            token_type_ids=token_type_ids,
        )

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
