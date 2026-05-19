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
from typing import List

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

from xpm_torch.configuration import FabricConfiguration
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.rankers.scorer import AbstractModuleScorer
from xpmir.text.encoders import TextEncoderBase
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
#   metadata.json    — dim, num_docs, n_bits
#

_PLAID_SUBDIR = "plaid"
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
    """Set of documents to index."""

    compress_only: Param[bool] = False

    index_path: Meta[Path]
    """Directory containing the fast-plaid index and side-car files."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device used to load the index for :meth:`get_document_tokens`
    (``""`` = auto). Fixed at first use because the underlying
    ``FastPlaid`` instance is cached."""

    in_memory: Meta[bool] = field(default=False, ignore_default=True)
    """If ``True``, load the index fully into device memory (passes
    ``low_memory=False`` to fast-plaid). Use when the index fits in
    VRAM/RAM and you want faster decompression/search; otherwise the
    document codes and residuals stay memory-mapped from disk."""

    def _plaid_dir(self) -> Path:
        return self.index_path / _PLAID_SUBDIR

    def _get_fast_plaid(self):
        """Return a cached ``FastPlaid`` instance for this index.

        The instance is constructed lazily on first access and reused
        afterwards so that subsequent calls (e.g. repeated
        :meth:`get_document_tokens`) avoid reloading the index.
        """
        cached = getattr(self, "_fast_plaid", None)
        if cached is not None:
            return cached
        fp_search = _import_fast_plaid()
        fp = fp_search.FastPlaid(
            index=str(self._plaid_dir()),
            device=self.device or None,
            low_memory=not self.in_memory,
        )
        self._fast_plaid = fp
        return fp

    def get_document_tokens(self, docid: int) -> torch.Tensor:
        """Return the (approximate) per-token embeddings for a document.

        The vectors are reconstructed from fast-plaid's compressed
        centroid + residual storage using ``FastPlaid.get_embeddings``.
        The reconstruction quality depends on :attr:`n_bits`.

        The underlying ``FastPlaid`` instance is cached (see
        :meth:`_get_fast_plaid`); its device and memory mode are
        controlled by :attr:`device` and :attr:`in_memory`.

        :param docid: Internal position of the document in the index
            (``0..num_docs-1``). External-to-internal mapping, if any,
            is the caller's responsibility.
        :returns: A ``(num_tokens, dim)`` float tensor containing the
            reconstructed token embeddings.
        """
        fp = self._get_fast_plaid()
        return fp.get_embeddings(subset=[int(docid)])[0]

    def get_documents_tokens(self, docids: List[int]) -> List[torch.Tensor]:
        """Return per-token embeddings for a batch of documents.

        Issues a single call to fast-plaid's ``get_embeddings``, which is
        materially faster than calling :meth:`get_document_tokens` in a
        loop.

        :param docids: Internal document positions
            (``0..num_docs-1``). External-to-internal mapping, if any,
            is the caller's responsibility.
        :returns: A list of ``(num_tokens, dim)`` float tensors, one per
            input id, in the same order.
        """
        if not docids:
            return []
        fp = self._get_fast_plaid()
        return fp.get_embeddings(subset=[int(d) for d in docids])


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

    encoder: Param[TextEncoderBase]
    """The ColBERT-style encoder used to produce per-token embeddings."""

    batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Encoder batch size. Warning, different from the batch size used internally by fast-plaid"""

    warmup_docs: Param[int] = field(default=1000, ignore_default=True)
    """Number of documents to encode and accumulate in RAM before creating the fast-plaid index and fitting the centroids.
    The token embeddings used to initialize the centroids will be sampled randomly from those documents by plaid
    (or they will all be used if n_samples_kmeans is 0)."""

    fast_plaid_batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Fast plaid internal batch size."""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations performed by fast-plaid when clustering
    the centroids."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    max_points_per_centroid: Param[int] = field(default=256, ignore_default=True)
    """Maximum number of points (documents) per centroid. Controls the creation of new centroids."""

    seed: Param[int] = field(default=42, ignore_default=True)
    """Random seed for reproducibility (passed to fast-plaid's index creation)."""

    compress_only: Param[bool] = field(default=False, ignore_default=True)
    """When ``True``, skip IVF construction. The resulting index supports
    :meth:`PlaidIndex.get_document_tokens` but not search via
    :class:`PlaidRetriever`.

    Requires fast-plaid support for ``compress_only``
    (see `lightonai/fast-plaid#41 <https://github.com/lightonai/fast-plaid/pull/41>`_).
    Falls back to building the full index with a warning if unsupported."""

    low_memory: Param[bool] = field(default=True)
    """https://github.com/lightonai/fast-plaid#-search-speed-tip-low_memoryfalse
    If index fits on VRAM, set to False for faster search. Otherwise, keep True to avoid OOM errors."""

    fabric_config: Meta[FabricConfiguration] = field(
        default_factory=FabricConfiguration.C
    )
    """Control the device for the model encoding and fast-plaid index."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Output directory for the index and its side-car files."""

    def task_outputs(self, dep) -> PlaidIndex:
        """Expose a :class:`PlaidIndex` for downstream tasks."""
        return dep(
            PlaidIndex.C(
                documents=self.documents,
                index_path=self.index_path,
                compress_only=self.compress_only,
            )
        )

    def execute(self):
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Fabric first
        fabric = self.fabric_config.get_fabric()
        fabric.launch()

        with fabric.init_module():
            self.encoder.initialize()
            self.encoder = fabric.setup(self.encoder)
            self.encoder.eval()

        fp_search = _import_fast_plaid()

        plaid_dir = self.index_path / _PLAID_SUBDIR
        plaid_dir.mkdir(parents=True, exist_ok=True)

        device = fabric.device or None
        fast_plaid = fp_search.FastPlaid(
            index=str(plaid_dir), device=device, low_memory=self.low_memory
        )

        total_docs = self.documents.documentcount or 0
        num_docs_seen = 0
        index_created = False
        warmup_buffer: list = []

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

                if not index_created:
                    warmup_buffer.extend(per_doc_cpu)

                    if num_docs_seen + len(per_doc_cpu) >= self.warmup_docs:
                        logging.info(
                            "Warmup buffer filled (%d documents, %d tokens). "
                            "Creating the fast-plaid index and fitting centroids...",
                            len(warmup_buffer),
                            sum(t.shape[0] for t in warmup_buffer),
                        )
                        # Enough docs accumulated — fit centroids and create index
                        create_kwargs = {
                            "documents_embeddings": warmup_buffer,
                            "nbits": self.n_bits,
                            "kmeans_niters": self.kmeans_niters,
                            "batch_size": self.fast_plaid_batch_size,
                            "seed": self.seed,
                            "max_points_per_centroid": self.max_points_per_centroid,
                        }
                        if self.n_samples_kmeans:
                            create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
                        if self.compress_only:
                            create_kwargs["compress_only"] = True
                        try:
                            fast_plaid.create(**create_kwargs)
                        except TypeError:
                            if self.compress_only:
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

                        warmup_buffer.clear()  # free RAM immediately
                        index_created = True
                else:
                    create_kwargs = {
                        "kmeans_niters": self.kmeans_niters,
                        "batch_size": self.fast_plaid_batch_size,
                        "seed": self.seed,
                        "max_points_per_centroid": self.max_points_per_centroid,
                    }
                    if self.n_samples_kmeans:
                        create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
                    if self.compress_only:
                        create_kwargs["compress_only"] = True

                    fast_plaid.update(documents_embeddings=per_doc_cpu, **create_kwargs)

                num_docs_seen += len(per_doc_cpu)
                pbar.update(len(per_doc_cpu))
            pbar.close()

        # In case the whole corpus was smaller than the warmup buffer, we still want to create the index
        if not index_created and warmup_buffer:
            create_kwargs = {
                "documents_embeddings": warmup_buffer,
                "nbits": self.n_bits,
                "kmeans_niters": self.kmeans_niters,
                "batch_size": self.fast_plaid_batch_size,
            }
            if self.n_samples_kmeans:
                create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
            fast_plaid.create(**create_kwargs)
            warmup_buffer.clear()

        with (self.index_path / _METADATA_FILE).open("w") as fh:
            json.dump(
                {
                    "num_docs": num_docs_seen,
                    "dim": self.encoder.dimension,
                    "n_bits": self.n_bits,
                },
                fh,
            )

        logger.info(
            "fast-plaid index built: %d documents, dim=%d, n_bits=%d",
            num_docs_seen,
            self.encoder.dimension,
            self.n_bits,
        )


class PlaidRetriever(Retriever):
    """Retriever using a `fast-plaid`_ PLAID index.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    encoder: Param[AbstractModuleScorer]
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

    fabric_config: Meta[FabricConfiguration] = field(
        default_factory=FabricConfiguration.C
    )
    """Control the device for the model encoding and fast-plaid index."""

    def initialize(self):
        super().initialize()
        if self.index.compress_only:
            raise RuntimeError(
                "Cannot search a compress-only PLAID index. "
                "Rebuild with compress_only=False to enable retrieval."
            )

        logger.info("PLAID retriever (1/2): initializing the encoder")
        # 1. Initialize Fabric first
        fabric = self.fabric_config.get_fabric()
        fabric.launch()

        with fabric.init_module():
            self.encoder.initialize()
            self.encoder = fabric.setup(self.encoder)
            self.encoder.eval()

        logger.info("PLAID retriever (2/2): opening the fast-plaid index")
        fp_search = _import_fast_plaid()
        device = fabric.device or None
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
