"""Interface to the fast-plaid library.

`fast-plaid <https://github.com/lightonai/fast-plaid>`_ is a Rust-based
implementation of PLAID / ColBERT late-interaction retrieval. This module
wraps it to build and query an index from a
:class:`~xpmir.neural.colbert.ColBERTEncoder`.

Two classes are exposed:

- :class:`PlaidIndex` — the index configuration (paths, metadata, accessor to
  the raw token embeddings).
- :class:`PlaidIndexBuilder` — a :class:`~experimaestro.Task` that encodes a
  :class:`~datamaestro_ir.data.DocumentStore` and builds the fast-plaid index.
- :class:`PlaidRetriever` — a :class:`~xpmir.rankers.Retriever` that searches
  the index given a query.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from experimaestro import (
    Config,
    Meta,
    Param,
    PathGenerator,
    Task,
    field,
    tqdm,
)
from datamaestro_ir.data import DocumentStore, IDTextRecord

from xpmir.neural.colbert import ColBERTEncoder
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.utils.utils import batchiter

logger = logging.getLogger(__name__)


def _import_fast_plaid():
    try:
        from fast_plaid import search as fp_search  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover - guard clause
        raise ModuleNotFoundError(
            "fast-plaid is not installed. Install it with `pip install fast-plaid`"
            " (see https://github.com/lightonai/fast-plaid)."
        ) from exc
    return fp_search


# File layout (under ``index_path``):
#
#   plaid/           — fast-plaid index directory
#   tokens.dat       — concatenated float32 per-token embeddings (optional)
#   offsets.npy      — int64 offsets into tokens.dat (optional)
#   ext2int.json     — external -> internal docid map (optional)
#   metadata.json    — dim, num_docs, store_tokens flag

_PLAID_SUBDIR = "plaid"
_TOKENS_FILE = "tokens.dat"
_OFFSETS_FILE = "offsets.npy"
_EXT2INT_FILE = "ext2int.json"
_METADATA_FILE = "metadata.json"


class PlaidIndex(Config):
    """A ColBERT / PLAID index backed by `fast-plaid`_.

    Besides the fast-plaid index itself, the index optionally stores the raw
    (unquantised) per-token document embeddings so that, given a document
    identifier, the full set of token vectors can be retrieved via
    :meth:`get_document_tokens`.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    documents: Param[DocumentStore]
    """The indexed document collection."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Directory containing the fast-plaid index and side-car files."""

    dim: Param[int]
    """Per-token embedding dimension stored in the index."""

    store_tokens: Param[bool] = field(default=True, ignore_default=True)
    """Whether raw per-token embeddings are kept on disk alongside the
    fast-plaid index (required by :meth:`get_document_tokens`)."""

    def _plaid_dir(self) -> Path:
        return self.index_path / _PLAID_SUBDIR

    def _tokens_file(self) -> Path:
        return self.index_path / _TOKENS_FILE

    def _offsets_file(self) -> Path:
        return self.index_path / _OFFSETS_FILE

    def _ext2int_file(self) -> Path:
        return self.index_path / _EXT2INT_FILE

    def _load_offsets(self) -> np.ndarray:
        offsets_path = self._offsets_file()
        if not offsets_path.exists():
            raise FileNotFoundError(
                f"No token offsets at {offsets_path}: the index was built "
                "without store_tokens=True"
            )
        return np.load(str(offsets_path))

    def _load_tokens_memmap(self, offsets: np.ndarray) -> np.memmap:
        tokens_path = self._tokens_file()
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"No raw tokens at {tokens_path}: the index was built "
                "without store_tokens=True"
            )
        total = int(offsets[-1])
        return np.memmap(
            str(tokens_path),
            dtype=np.float32,
            mode="r",
            shape=(total, self.dim),
        )

    def _load_ext2int(self) -> dict:
        path = self._ext2int_file()
        if not path.exists():
            return {}
        with path.open("r") as fh:
            return json.load(fh)

    def get_document_tokens(self, docid: Union[int, str]) -> torch.Tensor:
        """Return the per-token embeddings stored for a document.

        :param docid: The document identifier. Integers are interpreted as
            internal positions in the index (``0..num_docs-1``); strings are
            looked up in the external-to-internal map written at indexing
            time.
        :returns: A ``(num_tokens, dim)`` float tensor containing the
            (projected, normalised) token embeddings.
        :raises FileNotFoundError: if the index was built with
            ``store_tokens=False``.
        :raises KeyError: if the external identifier is unknown.
        """
        if isinstance(docid, str):
            ext2int = self._load_ext2int()
            if docid not in ext2int:
                raise KeyError(
                    f"External document id {docid!r} is unknown to this index"
                )
            internal = int(ext2int[docid])
        else:
            internal = int(docid)

        offsets = self._load_offsets()
        if internal < 0 or internal >= len(offsets) - 1:
            raise IndexError(
                f"Internal document id {internal} out of range [0, {len(offsets) - 1})"
            )

        tokens = self._load_tokens_memmap(offsets)
        start, end = int(offsets[internal]), int(offsets[internal + 1])
        # Copy out of the memmap so the returned tensor is safe to keep.
        return torch.from_numpy(np.asarray(tokens[start:end]).copy())


class PlaidIndexBuilder(Task):
    """Builds a fast-plaid index from a document collection.

    The builder encodes every document using the given
    :class:`~xpmir.neural.colbert.ColBERTEncoder`, collects the valid (i.e.
    non-padding) token vectors, and feeds them to ``fast-plaid``. When
    ``store_tokens`` is true, the raw per-token embeddings are additionally
    written as a memory-mapped float32 file so that they can be retrieved
    later via :meth:`PlaidIndex.get_document_tokens`.
    """

    documents: Param[DocumentStore]
    """Set of documents to index."""

    encoder: Param[ColBERTEncoder]
    """The ColBERT-style encoder used to produce per-token embeddings."""

    batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Encoder batch size."""

    store_tokens: Param[bool] = field(default=True, ignore_default=True)
    """Whether to store the raw per-token document embeddings. Required to
    later recover token vectors by document id."""

    n_bits: Meta[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Meta[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations performed by fast-plaid when clustering
    the centroids."""

    n_samples_kmeans: Meta[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for fast-plaid (``""`` = auto: cuda if available, cpu otherwise)."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Output directory for the index and its side-car files."""

    def task_outputs(self, dep) -> PlaidIndex:
        """Expose a :class:`PlaidIndex` for downstream tasks."""
        return dep(
            PlaidIndex.C(
                documents=self.documents,
                index_path=self.index_path,
                dim=self.encoder.dim,
                store_tokens=self.store_tokens,
            )
        )

    # ------------------------------------------------------------------ exec

    def _iter_doc_batches(self):
        doc_iter = self.documents.iter_documents()
        return batchiter(self.batch_size, doc_iter)

    def execute(self):
        # Clean up a stale output directory, then recreate it
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialise the encoder in eval mode (no gradients during indexing)
        self.encoder.initialize()
        self.encoder.eval()

        fp_search = _import_fast_plaid()

        plaid_dir = self.index_path / _PLAID_SUBDIR
        plaid_dir.mkdir(parents=True, exist_ok=True)

        device = self.device or None
        fast_plaid = fp_search.FastPlaid(index=str(plaid_dir), device=device)

        total_docs = self.documents.documentcount or 0

        # Raw-token storage buffers (only used when store_tokens is True)
        tokens_file = None
        offsets: List[int] = [0] if self.store_tokens else []
        ext2int: dict = {}
        dim = self.encoder.dim

        if self.store_tokens:
            tokens_path = self.index_path / _TOKENS_FILE
            tokens_file = tokens_path.open("wb")

        num_docs_seen = 0
        first_batch = True

        try:
            with torch.no_grad():
                pbar = tqdm(
                    total=total_docs or None,
                    desc="Encoding documents for fast-plaid",
                    unit="doc",
                )
                for batch in self._iter_doc_batches():
                    # Encode the batch to get one tensor per document, shape
                    # (num_valid_tokens, dim). Move to CPU float32 since
                    # fast-plaid (and our on-disk format) expect that.
                    per_doc = self.encoder.document_token_embeddings(batch)
                    per_doc_cpu = [
                        t.detach().to("cpu", dtype=torch.float32) for t in per_doc
                    ]

                    # Feed fast-plaid. Use create() on the first batch then
                    # update() on subsequent batches.
                    if first_batch:
                        fast_plaid.create(
                            documents_embeddings=per_doc_cpu,
                            n_bits=self.n_bits,
                            kmeans_niters=self.kmeans_niters,
                            **(
                                {"n_samples_kmeans": self.n_samples_kmeans}
                                if self.n_samples_kmeans
                                else {}
                            ),
                        )
                        first_batch = False
                    else:
                        fast_plaid.update(documents_embeddings=per_doc_cpu)

                    # Persist raw token embeddings if requested
                    if self.store_tokens:
                        for record, tensor in zip(batch, per_doc_cpu):
                            tokens_np = tensor.numpy().astype(np.float32, copy=False)
                            assert tokens_np.ndim == 2 and tokens_np.shape[1] == dim
                            tokens_file.write(tokens_np.tobytes(order="C"))
                            offsets.append(offsets[-1] + tokens_np.shape[0])
                            ext_id = (
                                record.get("id") if isinstance(record, dict) else None
                            )
                            if ext_id is not None:
                                ext2int[str(ext_id)] = num_docs_seen
                            num_docs_seen += 1
                    else:
                        num_docs_seen += len(per_doc_cpu)

                    pbar.update(len(per_doc_cpu))

                pbar.close()
        finally:
            if tokens_file is not None:
                tokens_file.close()

        if self.store_tokens:
            np.save(
                str(self.index_path / _OFFSETS_FILE),
                np.asarray(offsets, dtype=np.int64),
            )
            if ext2int:
                with (self.index_path / _EXT2INT_FILE).open("w") as fh:
                    json.dump(ext2int, fh)

        with (self.index_path / _METADATA_FILE).open("w") as fh:
            json.dump(
                {
                    "num_docs": num_docs_seen,
                    "dim": dim,
                    "store_tokens": bool(self.store_tokens),
                    "n_bits": self.n_bits,
                },
                fh,
            )

        logger.info(
            "fast-plaid index built: %d documents, dim=%d, store_tokens=%s",
            num_docs_seen,
            dim,
            self.store_tokens,
        )


class PlaidRetriever(Retriever):
    """Retriever using a `fast-plaid`_ PLAID index.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    encoder: Param[ColBERTEncoder]
    """The query encoder. Typically the same encoder that was used to build
    :attr:`index`."""

    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    topk: Param[int]
    """Number of documents to return per query."""

    n_ivf_probe: Meta[int] = field(default=8, ignore_default=True)
    """Number of inverted-list clusters explored by fast-plaid at search time."""

    n_full_scores: Meta[int] = field(default=0, ignore_default=True)
    """Number of candidates for which fast-plaid computes full scores
    (0 = fast-plaid default)."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for fast-plaid (``""`` = auto)."""

    def initialize(self):
        super().initialize()
        logger.info("PLAID retriever (1/2): initializing the encoder")
        self.encoder.initialize()
        self.encoder.eval()

        logger.info("PLAID retriever (2/2): opening the fast-plaid index")
        fp_search = _import_fast_plaid()
        device = self.device or None
        self._fast_plaid = fp_search.FastPlaid(
            index=str(self.index._plaid_dir()), device=device
        )

    def _store(self):
        return self.index.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        with torch.no_grad():
            # ColBERTEncoder.query_token_embeddings returns (1, L, dim)
            queries_embeddings = self.encoder.query_token_embeddings([record])
            queries_embeddings = queries_embeddings.detach().to(
                "cpu", dtype=torch.float32
            )

            search_kwargs = {"top_k": self.topk, "n_ivf_probe": self.n_ivf_probe}
            if self.n_full_scores:
                search_kwargs["n_full_scores"] = self.n_full_scores

            results = self._fast_plaid.search(
                queries_embeddings=queries_embeddings,
                **search_kwargs,
            )

        # fast-plaid returns a list (one entry per query) of ranked
        # ``(doc_index, score)`` tuples.
        single = results[0] if results else []
        documents = self.index.documents
        out: List[ScoredDocument] = []
        for doc_index, score in single:
            out.append(
                ScoredDocument(documents.document_int(int(doc_index)), float(score))
            )
        return out
