"""Interface to the fast-plaid library.

`fast-plaid <https://github.com/lightonai/fast-plaid>`_ is a Rust-based
implementation of PLAID / ColBERT late-interaction retrieval. This module
wraps it to build and query an index from a
:class:`~xpmir.neural.colbert.ColBERTEncoder`.

The module is split in two layers that can be used independently:

- **Token store** (:class:`TokenStore` / :class:`TokenStoreBuilder`):
  encodes every document using a ColBERT encoder and persists the raw
  (unquantised) per-token embeddings on disk. No fast-plaid dependency is
  needed. The store can retrieve per-document token vectors by internal
  integer id or external string id via :meth:`TokenStore.get_document_tokens`.

- **PLAID index** (:class:`PlaidIndex` / :class:`PlaidIndexBuilder` /
  :class:`PlaidRetriever`): builds the fast-plaid centroid/IVF structure
  *from* a pre-built token store and exposes a standard
  :class:`~xpmir.rankers.Retriever` interface for approximate search.
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


# ---- File layout -----------------------------------------------------------
#
# Token store (under ``store_path``):
#   tokens.dat       — concatenated float32 per-token embeddings
#   offsets.npy      — int64 offsets into tokens.dat
#   ext2int.json     — external -> internal docid map
#   metadata.json    — dim, num_docs
#
# PLAID index (under ``index_path``):
#   plaid/           — fast-plaid index directory

_TOKENS_FILE = "tokens.dat"
_OFFSETS_FILE = "offsets.npy"
_EXT2INT_FILE = "ext2int.json"
_METADATA_FILE = "metadata.json"
_PLAID_SUBDIR = "plaid"


# ---------------------------------------------------------------------------
# Token store — no fast-plaid dependency
# ---------------------------------------------------------------------------


class TokenStore(Config):
    """On-disk store of per-token document embeddings.

    Given a document identifier the store returns the ``(num_tokens, dim)``
    tensor of (projected, normalised) token vectors that were produced by
    the ColBERT encoder at indexing time.

    The store is a flat file pair:

    * ``tokens.dat`` — a concatenated float32 array of all token vectors.
    * ``offsets.npy`` — an int64 array of length ``num_docs + 1`` giving
      the start offset for each document into ``tokens.dat``.

    No fast-plaid dependency is required.
    """

    documents: Param[DocumentStore]
    """The indexed document collection."""

    store_path: Meta[Path] = field(default_factory=PathGenerator("token-store"))
    """Directory containing the token files."""

    dim: Param[int]
    """Per-token embedding dimension."""

    # -- internal helpers ---------------------------------------------------

    def _tokens_file(self) -> Path:
        return self.store_path / _TOKENS_FILE

    def _offsets_file(self) -> Path:
        return self.store_path / _OFFSETS_FILE

    def _ext2int_file(self) -> Path:
        return self.store_path / _EXT2INT_FILE

    def _load_offsets(self) -> np.ndarray:
        return np.load(str(self._offsets_file()))

    def _load_tokens_memmap(self, offsets: np.ndarray) -> np.memmap:
        total = int(offsets[-1])
        return np.memmap(
            str(self._tokens_file()),
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

    # -- public API ---------------------------------------------------------

    def get_document_tokens(self, docid: Union[int, str]) -> torch.Tensor:
        """Return the per-token embeddings stored for a document.

        :param docid: The document identifier. Integers are interpreted as
            internal positions in the store (``0..num_docs-1``); strings are
            looked up in the external-to-internal map written at indexing
            time.
        :returns: A ``(num_tokens, dim)`` float tensor.
        :raises KeyError: if the external identifier is unknown.
        :raises IndexError: if the internal id is out of range.
        """
        if isinstance(docid, str):
            ext2int = self._load_ext2int()
            if docid not in ext2int:
                raise KeyError(
                    f"External document id {docid!r} is unknown to this store"
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
        return torch.from_numpy(np.asarray(tokens[start:end]).copy())


class TokenStoreBuilder(Task):
    """Encodes a document collection and writes a :class:`TokenStore`.

    For each document the encoder produces one ``(num_valid_tokens, dim)``
    tensor. The tensors are written contiguously into ``tokens.dat`` and
    their boundaries recorded in ``offsets.npy``.

    This task does **not** require fast-plaid.
    """

    documents: Param[DocumentStore]
    """Set of documents to encode."""

    encoder: Param[ColBERTEncoder]
    """The ColBERT-style encoder used to produce per-token embeddings."""

    batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Encoder batch size."""

    store_path: Meta[Path] = field(default_factory=PathGenerator("token-store"))
    """Output directory for the token store files."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for the encoder (``""`` = auto)."""

    def task_outputs(self, dep) -> TokenStore:
        return dep(
            TokenStore.C(
                documents=self.documents,
                store_path=self.store_path,
                dim=self.encoder.dim,
            )
        )

    def execute(self):
        if self.store_path.exists():
            shutil.rmtree(self.store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.encoder.initialize()
        self.encoder.eval()

        dim = self.encoder.dim
        total_docs = self.documents.documentcount or 0
        offsets: List[int] = [0]
        ext2int: dict = {}
        num_docs_seen = 0

        tokens_path = self.store_path / _TOKENS_FILE
        tokens_file = tokens_path.open("wb")

        try:
            with torch.no_grad():
                pbar = tqdm(
                    total=total_docs or None,
                    desc="Encoding documents",
                    unit="doc",
                )
                for batch in batchiter(
                    self.batch_size, self.documents.iter_documents()
                ):
                    per_doc = self.encoder.document_token_embeddings(batch)
                    for record, tensor in zip(batch, per_doc):
                        arr = (
                            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                        assert arr.ndim == 2 and arr.shape[1] == dim
                        tokens_file.write(arr.tobytes(order="C"))
                        offsets.append(offsets[-1] + arr.shape[0])
                        ext_id = record.get("id") if isinstance(record, dict) else None
                        if ext_id is not None:
                            ext2int[str(ext_id)] = num_docs_seen
                        num_docs_seen += 1
                    pbar.update(len(per_doc))
                pbar.close()
        finally:
            tokens_file.close()

        np.save(
            str(self.store_path / _OFFSETS_FILE),
            np.asarray(offsets, dtype=np.int64),
        )
        if ext2int:
            with (self.store_path / _EXT2INT_FILE).open("w") as fh:
                json.dump(ext2int, fh)
        with (self.store_path / _METADATA_FILE).open("w") as fh:
            json.dump({"num_docs": num_docs_seen, "dim": dim}, fh)

        logger.info("Token store built: %d documents, dim=%d", num_docs_seen, dim)


# ---------------------------------------------------------------------------
# PLAID index — requires fast-plaid
# ---------------------------------------------------------------------------


class PlaidIndex(Config):
    """A ColBERT / PLAID search index backed by `fast-plaid`_.

    Wraps a :class:`TokenStore` (the raw per-token embeddings) and the
    fast-plaid centroid/IVF structure (built by :class:`PlaidIndexBuilder`).

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    token_store: Param[TokenStore]
    """The underlying token store (also reachable for ``get_document_tokens``)."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Directory containing the fast-plaid index files."""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations used to build the centroids."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    def _plaid_dir(self) -> Path:
        return self.index_path / _PLAID_SUBDIR


class PlaidIndexBuilder(Task):
    """Builds a fast-plaid search index from an existing :class:`TokenStore`.

    The builder reads the raw token embeddings from the store and feeds them
    to fast-plaid. This step is separate from :class:`TokenStoreBuilder` so
    that the token store can be used independently (e.g. to retrieve
    per-document token vectors) without paying the cost of building the
    centroid/IVF structure.
    """

    token_store: Param[TokenStore]
    """A pre-built token store."""

    batch_size: Meta[int] = field(default=1024, ignore_default=True)
    """Number of documents read from the store at a time."""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations for centroid training."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for fast-plaid (``""`` = auto: cuda if available,
    cpu otherwise)."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Output directory for the fast-plaid index files."""

    def task_outputs(self, dep) -> PlaidIndex:
        return dep(
            PlaidIndex.C(
                token_store=self.token_store,
                index_path=self.index_path,
                n_bits=self.n_bits,
                kmeans_niters=self.kmeans_niters,
                n_samples_kmeans=self.n_samples_kmeans,
            )
        )

    def execute(self):
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        fp_search = _import_fast_plaid()

        plaid_dir = self.index_path / _PLAID_SUBDIR
        plaid_dir.mkdir(parents=True, exist_ok=True)

        device = self.device or None
        fast_plaid = fp_search.FastPlaid(index=str(plaid_dir), device=device)

        # Load the token store as a memmap
        store = self.token_store
        offsets = store._load_offsets()
        tokens_mmap = store._load_tokens_memmap(offsets)
        num_docs = len(offsets) - 1

        first_batch = True
        pbar = tqdm(total=num_docs, desc="Building fast-plaid index", unit="doc")

        for batch_start in range(0, num_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_docs)
            per_doc: List[torch.Tensor] = []
            for i in range(batch_start, batch_end):
                start, end = int(offsets[i]), int(offsets[i + 1])
                t = torch.from_numpy(np.asarray(tokens_mmap[start:end]).copy())
                per_doc.append(t)

            if first_batch:
                create_kwargs = {
                    "documents_embeddings": per_doc,
                    "n_bits": self.n_bits,
                    "kmeans_niters": self.kmeans_niters,
                }
                if self.n_samples_kmeans:
                    create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
                fast_plaid.create(**create_kwargs)
                first_batch = False
            else:
                fast_plaid.update(documents_embeddings=per_doc)

            pbar.update(batch_end - batch_start)

        pbar.close()
        logger.info("fast-plaid index built (%d documents)", num_docs)


class PlaidRetriever(Retriever):
    """Retriever using a `fast-plaid`_ PLAID index.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    encoder: Param[ColBERTEncoder]
    """The query encoder. Typically the same encoder that was used to build
    the token store."""

    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    topk: Param[int]
    """Number of documents to return per query."""

    n_ivf_probe: Meta[int] = field(default=8, ignore_default=True)
    """Number of inverted-list clusters explored at search time."""

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
        return self.index.token_store.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        with torch.no_grad():
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

        single = results[0] if results else []
        documents = self.index.token_store.documents
        out: List[ScoredDocument] = []
        for doc_index, score in single:
            out.append(
                ScoredDocument(documents.document_int(int(doc_index)), float(score))
            )
        return out
