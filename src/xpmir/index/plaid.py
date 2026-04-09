"""Interface to the fast-plaid library.

`fast-plaid <https://github.com/lightonai/fast-plaid>`_ is a Rust-based
implementation of PLAID / ColBERT late-interaction retrieval. This module
wraps it to build and query an index from a
:class:`~xpmir.neural.colbert.ColBERTEncoder`.

Three classes are exposed:

- :class:`PlaidIndex` — the index configuration (paths, metadata). Supports
  retrieving per-document token vectors via :meth:`PlaidIndex.get_document_tokens`
  using fast-plaid's compressed centroid+residual storage.
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
#   ext2int.json     — external -> internal docid map
#   metadata.json    — dim, num_docs, n_bits

_PLAID_SUBDIR = "plaid"
_EXT2INT_FILE = "ext2int.json"
_METADATA_FILE = "metadata.json"


class PlaidIndex(Config):
    """A ColBERT / PLAID index backed by `fast-plaid`_.

    The index stores per-token document embeddings in fast-plaid's compressed
    centroid + residual format. Per-document token vectors can be
    reconstructed (approximately) via :meth:`get_document_tokens`, which
    delegates to fast-plaid's ``get_embeddings`` method. The reconstruction
    quality is controlled by :attr:`n_bits`.

    When :attr:`compress_only` is ``True`` the index only contains the
    compressed vectors (centroids + quantised residuals) without the IVF
    search structure. This is cheaper to build and sufficient when only
    :meth:`get_document_tokens` is needed. Attempting to search a
    compress-only index via :class:`PlaidRetriever` will raise an error.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    documents: Param[DocumentStore]
    """The indexed document collection."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Directory containing the fast-plaid index and side-car files."""

    dim: Param[int]
    """Per-token embedding dimension stored in the index."""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations used to build the centroids."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    compress_only: Param[bool] = field(default=False, ignore_default=True)
    """When ``True`` the fast-plaid index is built without the IVF search
    structure. This skips the expensive inverted-list construction and is
    sufficient when only :meth:`get_document_tokens` is needed.

    Requires fast-plaid support for ``compress_only``
    (see `lightonai/fast-plaid#41 <https://github.com/lightonai/fast-plaid/pull/41>`_).
    If the installed version does not support it, the full index is built
    instead and a warning is logged."""

    def _plaid_dir(self) -> Path:
        return self.index_path / _PLAID_SUBDIR

    def _ext2int_file(self) -> Path:
        return self.index_path / _EXT2INT_FILE

    def _load_ext2int(self) -> dict:
        path = self._ext2int_file()
        if not path.exists():
            return {}
        with path.open("r") as fh:
            return json.load(fh)

    def get_document_tokens(
        self,
        docid: Union[int, str],
        device: str = "",
    ) -> torch.Tensor:
        """Return the (approximate) per-token embeddings for a document.

        The vectors are reconstructed from fast-plaid's compressed
        centroid + residual storage using ``FastPlaid.get_embeddings``.
        The reconstruction quality depends on :attr:`n_bits`.

        :param docid: The document identifier. Integers are interpreted as
            internal positions in the index (``0..num_docs-1``); strings are
            looked up in the external-to-internal map written at indexing
            time.
        :param device: Device for the fast-plaid instance used to decompress
            (``""`` = auto).
        :returns: A ``(num_tokens, dim)`` float tensor containing the
            reconstructed token embeddings.
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

        fp_search = _import_fast_plaid()
        fp = fp_search.FastPlaid(index=str(self._plaid_dir()), device=device or None)
        results = fp.get_embeddings(subset=[internal])
        return results[0]


class PlaidIndexBuilder(Task):
    """Builds a fast-plaid index from a document collection.

    The builder encodes every document using the given
    :class:`~xpmir.neural.colbert.ColBERTEncoder`, collects the valid (i.e.
    non-padding) token vectors, and feeds them to ``fast-plaid``.

    The fast-plaid index stores the embeddings in a compressed
    centroid + residual format, so no separate raw-token file is needed.
    Per-document token vectors can be reconstructed later via
    :meth:`PlaidIndex.get_document_tokens`.
    """

    documents: Param[DocumentStore]
    """Set of documents to index."""

    encoder: Param[ColBERTEncoder]
    """The ColBERT-style encoder used to produce per-token embeddings."""

    batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Encoder batch size."""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations performed by fast-plaid when clustering
    the centroids."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    compress_only: Param[bool] = field(default=False, ignore_default=True)
    """When ``True``, skip IVF construction. The resulting index supports
    :meth:`PlaidIndex.get_document_tokens` but not search via
    :class:`PlaidRetriever`.

    Requires fast-plaid support for ``compress_only``
    (see `lightonai/fast-plaid#41 <https://github.com/lightonai/fast-plaid/pull/41>`_).
    Falls back to building the full index with a warning if unsupported."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for fast-plaid (``""`` = auto: cuda if available,
    cpu otherwise)."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Output directory for the index and its side-car files."""

    def task_outputs(self, dep) -> PlaidIndex:
        """Expose a :class:`PlaidIndex` for downstream tasks."""
        return dep(
            PlaidIndex.C(
                documents=self.documents,
                index_path=self.index_path,
                dim=self.encoder.dim,
                n_bits=self.n_bits,
                kmeans_niters=self.kmeans_niters,
                n_samples_kmeans=self.n_samples_kmeans,
                compress_only=self.compress_only,
            )
        )

    def execute(self):
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.encoder.initialize()
        self.encoder.eval()

        fp_search = _import_fast_plaid()

        plaid_dir = self.index_path / _PLAID_SUBDIR
        plaid_dir.mkdir(parents=True, exist_ok=True)

        device = self.device or None
        fast_plaid = fp_search.FastPlaid(index=str(plaid_dir), device=device)

        total_docs = self.documents.documentcount or 0
        ext2int: dict = {}
        num_docs_seen = 0
        first_batch = True

        with torch.no_grad():
            pbar = tqdm(
                total=total_docs or None,
                desc="Encoding documents for fast-plaid",
                unit="doc",
            )
            for batch in batchiter(self.batch_size, self.documents.iter_documents()):
                per_doc = self.encoder.document_token_embeddings(batch)
                per_doc_cpu = [
                    t.detach().to("cpu", dtype=torch.float32) for t in per_doc
                ]

                if first_batch:
                    create_kwargs = {
                        "documents_embeddings": per_doc_cpu,
                        "n_bits": self.n_bits,
                        "kmeans_niters": self.kmeans_niters,
                    }
                    if self.n_samples_kmeans:
                        create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
                    if self.compress_only:
                        create_kwargs["compress_only"] = True
                    try:
                        fast_plaid.create(**create_kwargs)
                    except TypeError:
                        if self.compress_only:
                            # compress_only not yet supported by installed
                            # fast-plaid — see
                            # https://github.com/lightonai/fast-plaid/pull/41
                            logger.warning(
                                "compress_only is not supported by this "
                                "version of fast-plaid; building the full "
                                "index instead. See "
                                "https://github.com/lightonai/fast-plaid/pull/41"
                            )
                            del create_kwargs["compress_only"]
                            fast_plaid.create(**create_kwargs)
                        else:
                            raise
                    first_batch = False
                else:
                    fast_plaid.update(documents_embeddings=per_doc_cpu)

                for record in batch:
                    ext_id = record.get("id") if isinstance(record, dict) else None
                    if ext_id is not None:
                        ext2int[str(ext_id)] = num_docs_seen
                    num_docs_seen += 1

                pbar.update(len(per_doc_cpu))

            pbar.close()

        if ext2int:
            with (self.index_path / _EXT2INT_FILE).open("w") as fh:
                json.dump(ext2int, fh)

        with (self.index_path / _METADATA_FILE).open("w") as fh:
            json.dump(
                {
                    "num_docs": num_docs_seen,
                    "dim": self.encoder.dim,
                    "n_bits": self.n_bits,
                },
                fh,
            )

        logger.info(
            "fast-plaid index built: %d documents, dim=%d, n_bits=%d",
            num_docs_seen,
            self.encoder.dim,
            self.n_bits,
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
    """Number of inverted-list clusters explored by fast-plaid at search
    time."""

    n_full_scores: Meta[int] = field(default=0, ignore_default=True)
    """Number of candidates for which fast-plaid computes full scores
    (0 = fast-plaid default)."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device for fast-plaid (``""`` = auto)."""

    def initialize(self):
        super().initialize()
        if self.index.compress_only:
            raise RuntimeError(
                "Cannot search a compress-only PLAID index. "
                "Rebuild with compress_only=False to enable retrieval."
            )

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
        documents = self.index.documents
        out: List[ScoredDocument] = []
        for doc_index, score in single:
            out.append(
                ScoredDocument(documents.document_int(int(doc_index)), float(score))
            )
        return out
