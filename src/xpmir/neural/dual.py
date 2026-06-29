from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any
from attrs import evolve
import torch
from experimaestro import DataPath, OptionalDataPath, field, Param
from datamaestro_ir.data import IDTextRecord
from xpmir.neural import DualRepresentationScorer, QueriesRep, DocsRep

from xpmir.text.encoders import TextEncoderBase
from xpm_torch.learner import TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.module import ModuleLoader
from xpm_torch.trainers import TrainingHook
from xpm_torch.metrics import ScalarMetric

import logging

logger = logging.getLogger(__name__)


class DualVectorListener(TrainingHook):
    """Listener called with the (vectorial) representation of queries and
    documents

    The hook is called just after the computation of documents and queries
    representations.

    This can be used for logging purposes, but more importantly, to add
    regularization losses such as the :class:`FlopsRegularizer` regularizer.
    """

    def __call__(
        self, context: TrainerContext, queries: torch.Tensor, documents: torch.Tensor
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context
            queries (torch.Tensor): The query vectors
            documents (torch.Tensor): The document vectors

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualVectorScorerListener(TrainingHook, ABC):
    """Listener called with the (vectorial) representation of queries and
    documents

    The hook is called just after the computation of documents and queries
    representations.

    This can be used for logging purposes, but more importantly, to add
    regularization losses such as the :class:`FlopsRegularizer` regularizer.
    """

    @abstractmethod
    def __call__(
        self,
        context: TrainerContext,
        queries: torch.Tensor,
        documents: torch.Tensor,
        scorer: torch.Tensor,
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context queries
            (torch.Tensor): The query vectors documents (torch.Tensor): The
            document vectors scores (torch.Tensor): A vector or matrix of scores
            (depending on how the scores were computed)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualModuleLoader(ModuleLoader):
    """ModuleLoader for dual encoder models.

    Has distinct ``encoder_path`` and ``query_encoder_path`` DataPaths so each
    encoder is serialized independently. This enables proper
    sentence-transformers format on HF Hub export (symmetric vs
    router/asymmetric).
    """

    encoder_path: Param[DataPath]
    """Path to the document encoder checkpoint"""

    query_encoder_path: OptionalDataPath = None
    """Path to the query encoder checkpoint (if separate from doc encoder)"""

    def execute(self):
        self.value.initialize()
        self.value.encoder.load_model(Path(self.encoder_path))
        if self.query_encoder_path and self.value.query_encoder is not None:
            self.value._query_encoder.load_model(Path(self.query_encoder_path))


class DualVectorScorer(DualRepresentationScorer[QueriesRep, DocsRep]):
    """A scorer based on dual vectorial representations"""

    CONFIG_LOADER = DualModuleLoader.C

    encoder: Param[TextEncoderBase[Any, Any]]
    """The document (and potentially query) encoder"""

    query_encoder: Param[Optional[TextEncoderBase[Any, Any]]]
    """The query encoder (optional, if not defined uses the query_encoder)"""

    def __initialize__(self):
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

    @property
    def _query_encoder(self):
        return self.query_encoder or self.encoder

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _has_separate_query_model(self) -> bool:
        """Check if query and doc encoders are separate models.
        Default: object identity check. Override for more specific logic.
        """
        return self.query_encoder is not None and self.query_encoder is not self.encoder

    def export_action(self, loader, **kwargs):
        from xpmir.models import XPMIRExportAction

        if self.doc:
            kwargs.setdefault("doc", self.doc)
        if self.bibtex:
            kwargs.setdefault("bibtex", self.bibtex)
        return XPMIRExportAction.C(loader=loader, **kwargs)

    def loader_config(self, path: Path, *, settings=None) -> DualModuleLoader:
        has_separate_query = self._has_separate_query_model()
        return self.CONFIG_LOADER(
            value=self,
            encoder_path=path / "encoder",
            query_encoder_path=(path / "query_encoder" if has_separate_query else None),
            settings=settings,
        )


class Dense(DualVectorScorer[QueriesRep, DocsRep]):
    """A scorer based on a pair of (query, document) dense vectors"""

    def score_product(self, queries, documents, info: Optional[TrainerContext] = None):
        # Gather the representations across all processes
        if info is not None and info.fabric is not None:
            fabric = info.fabric
            q_all = fabric.all_gather(queries.value, sync_grads=True)
            d_all = fabric.all_gather(documents.value, sync_grads=True)
        else:
            # We should always have access to fabric (even not in trainer mode)
            q_all = queries.value
            d_all = documents.value

        scores = q_all @ d_all.T

        if info is not None:
            for hook in info.hooks(DualVectorListener):
                hook(info, queries, documents)
            for hook in info.hooks(DualVectorScorerListener):
                hook(info, queries, documents, scores)

        return scores

    def score_pairs(self, queries, documents, info: Optional[TrainerContext] = None):
        scores = (
            (queries.value.unsqueeze(1) @ documents.value.unsqueeze(2))
            .squeeze(-1)
            .squeeze(-1)
        )

        # Apply the dual vector hook
        if info is not None:
            for hook in info.hooks(DualVectorListener):
                hook(info, queries, documents)
            for hook in info.hooks(DualVectorScorerListener):
                hook(info, queries, documents, scores)
        return scores

    @classmethod
    def from_pretrained_hf(cls, model_id: str, **kwargs):
        """Creates a Dense model from a HuggingFace encoder model

        :param model_id: The HuggingFace model ID
        :param kwargs: Additional keyword arguments passed to the constructor
        :returns: (model, init_tasks) tuple
        """
        from xpmir.text.huggingface.base import (
            HFConfigID,
            HFModel,
            HFModelInitFromID,
        )
        from xpmir.text.huggingface.encoders import HFCLSEncoder

        hf_model = HFModel.C(config=HFConfigID.C(hf_id=model_id))
        init_hf = HFModelInitFromID.C(model=hf_model)
        encoder = HFCLSEncoder.C(model=hf_model, **kwargs)
        return cls.C(encoder=encoder), [init_hf]

    @classmethod
    def from_sentence_transformers(cls, hf_id: str, **kwargs):
        """Creates a dense model from a Sentence transformer

        The list can be found on HuggingFace
        https://huggingface.co/models?library=sentence-transformers

        :param hf_id: The HuggingFace ID
        """
        from xpmir.text.huggingface.encoders import SentenceTransformerTextEncoder

        encoder = SentenceTransformerTextEncoder.C(model_id=hf_id)
        return cls.C(encoder=encoder, **kwargs)


class CosineDense(Dense):
    """Dual model based on cosine similarity."""

    def encode_queries(self, records: List[IDTextRecord]):
        queries = (self.query_encoder or self.encoder)(records)
        return evolve(
            queries, value=queries.value / queries.value.norm(dim=-1, keepdim=True)
        )

    def encode_documents(self, records: List[IDTextRecord]):
        documents = self.encoder(records)
        return evolve(
            documents,
            value=documents.value / documents.value.norm(dim=-1, keepdim=True),
        )


class DotDense(Dense):
    """Dual model based on inner product."""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def encode_queries(self, records: List[IDTextRecord]):
        """Encode the different queries"""
        return self._query_encoder(records)

    def encode_documents(self, records: List[IDTextRecord]):
        """Encode the different documents"""
        return self.encoder(records)

    def save_model(self, path: Path):
        """Save sub-encoders independently to subdirectories."""
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_model(path / "encoder")
        if self.query_encoder is not None and self.query_encoder is not self.encoder:
            self._query_encoder.save_model(path / "query_encoder")

    def load_model(self, path: Path):
        """Load sub-encoders from subdirectories, with backward compat."""
        if (path / "encoder").exists():
            self.encoder.load_model(path / "encoder")
            if (path / "query_encoder").exists() and self.query_encoder is not None:
                self._query_encoder.load_model(path / "query_encoder")
        else:
            # Backward compat: flat state_dict
            super().load_model(path)


def dual_representation_metrics(
    info: TrainerContext, queries: torch.Tensor, documents: torch.Tensor, compute_flops: bool = False
):
    """Compute and report diagnostic metrics for dual sparse representations.

    Covers:
    - Sparsity and average active-term length
    - Top-K energy concentration
    - QD-FLOPS estimates
    - Saturation of the most frequent terms
    - F_base and F_nce term-contribution metrics (when relevance labels are
      available via ``info.current_relevances``, set by BatchwiseTrainer)
    """
    with torch.no_grad():
        B_q = queries.shape[0]
        B_d = documents.shape[0]

        # ------------------------------------------------------------------
        # Sparsity: fraction of non-zero entries
        # ------------------------------------------------------------------
        info.metrics.add(ScalarMetric(
            "sparsity_q",
            torch.count_nonzero(queries).item() / (B_q * queries.shape[1]),
            B_q,
        ))
        info.metrics.add(ScalarMetric(
            "sparsity_d",
            torch.count_nonzero(documents).item() / (B_d * documents.shape[1]),
            B_d,
        ))

        # ------------------------------------------------------------------
        # Average active-term length per sample
        # ------------------------------------------------------------------
        info.metrics.add(ScalarMetric(
            "l_avg_q", (queries > 0).float().sum(dim=1).mean().item(), B_q
        ))
        info.metrics.add(ScalarMetric(
            "l_avg_d", (documents > 0).float().sum(dim=1).mean().item(), B_d
        ))

        # ------------------------------------------------------------------
        # Energy concentration: share of total activation mass in the top-K
        # terms (mean over the batch).
        # ------------------------------------------------------------------
        for prefix, x, k_list in [("q", queries, [1, 5, 10]), ("d", documents, [1, 25, 50])]:
            denom = x.sum(dim=1).clamp(min=1e-8)
            for k in k_list:
                k_clamped = min(k, x.shape[1])
                energy = torch.topk(x, k=k_clamped, dim=1).values.sum(dim=1) / denom
                info.metrics.add(ScalarMetric(f"energy_{prefix}/top{k}", energy.mean().item(), x.shape[0]))

        # ------------------------------------------------------------------
        # QD-FLOPS: expected dot-product operations
        #   qdflops_count — boolean co-activation rate (normalised)
        #   flops_dq      — weighted estimate (mean_q · mean_d)
        # ------------------------------------------------------------------
        qdflops_count = (
            ((queries > 0).sum(0) * (documents > 0).sum(0)).float().mean()
        ) / queries.shape[1]
        info.metrics.add(ScalarMetric("qdflops_count", qdflops_count.item(), 1))

        flops_dq = (queries.mean(dim=0) * documents.mean(dim=0)).sum()
        info.metrics.add(ScalarMetric("flops_dq", flops_dq.item(), 1))

        # ------------------------------------------------------------------
        # Saturation: median activation rate of the top-K most frequent terms
        # ------------------------------------------------------------------
        for prefix, x in [("q", queries), ("d", documents)]:
            freq = (x > 0).float().mean(0)
            top_freq = freq.topk(min(20, freq.shape[0])).values
            for k in (1, 5, 10, 20):
                if k <= top_freq.shape[0]:
                    info.metrics.add(ScalarMetric(
                        f"saturation_{prefix}/median_top{k}",
                        top_freq[:k].median().item(),
                        1,
                    ))

                    info.metrics.add(ScalarMetric(
                        f"saturation_{prefix}/mean_top{k}",
                        top_freq[:k].mean().item(),
                        1,
                    ))

            # How many dimensions are always on (fire in 100% of samples)
            info.metrics.add(ScalarMetric(
                f"saturation_{prefix}/always_on",
                (freq == 1.0).sum().item(),
                1,
            ))

            # Overall sparsity: fraction of vocab activated in this batch
            info.metrics.add(ScalarMetric(
                f"saturation_{prefix}/vocab_coverage",
                (freq > 0).float().mean().item(),
                1,
            ))



        # ------------------------------------------------------------------
        # F_base and F_nce — term-level contribution metrics.
        # Requires info.current_relevances (set by BatchwiseTrainer).
        # Silently skipped during inference or distillation training.
        #
        # F_base(i) = q_i·d+_i / Σ_j q_j·d+_j
        #   share of the positive dot-product carried by term i
        #
        # F_nce(i)  = q_i·d+_i / (q_i · Σ_{all docs} d_i)
        #   how exclusively term i discriminates the positive from the batch
        # ------------------------------------------------------------------
        rel = getattr(info, "current_relevances", None)

        if rel == "pairwise":
            # ------------------------------------------------------------------
            # Pairwise distillation layout (DistillationPairwiseTrainer).
            #
            # score_pairs is called with indexed tensors, so the hook receives:
            #   queries   (2N, V) — each query duplicated: [q0, q0, q1, q1, …]
            #   documents (2N, V) — interleaved pos/neg:   [d0+, d0-, d1+, d1-, …]
            #
            # The positive is unambiguous (it's always the even-indexed entry),
            # unlike the batchwise case which needs the relevance matrix.
            # ------------------------------------------------------------------
            # Layout is proved by tracing PairwiseItems:
            #   unique_documents = list(chain(positives, negatives))
            #                    → [pos0,…,posN-1, neg0,…,negN-1]  (block, not interleaved)
            #   pairs() → q_ix = [0,…,N-1, 0,…,N-1],  d_ix = [0,…,2N-1]
            #   score_pairs receives enc_queries[q_ix] and enc_documents[d_ix]
            #   → queries  = [q0,…,qN-1,   q0,…,qN-1]   (first half == second half)
            #   → documents= [pos0,…,posN-1, neg0,…,negN-1]
            N         = queries.shape[0] // 2
            queries_u = queries[:N]      # (N, V) — unique queries (first or second half, identical)
            d_pos     = documents[:N]    # (N, V) — all positives (first block)
            d_neg     = documents[N:]    # (N, V) — all negatives (second block)

            term_contrib  = queries_u * d_pos          # (N, V)
            mask          = term_contrib > 0            # (N, V)
            n_contributing = mask.sum().clamp(min=1)

            # F_base: share of the positive dot-product carried by each term
            total_score = term_contrib.sum(dim=1, keepdim=True).clamp(min=1e-8)
            f_base = term_contrib / total_score
            info.metrics.add(ScalarMetric(
                "f_base", f_base[mask].mean().item(), n_contributing.item()
            ))

            # F_nce: how exclusively each term points to d+ vs. all in-batch docs
            # d_sum includes both positives and negatives, approximating corpus sum
            d_sum     = documents.sum(dim=0)                              # (V,)
            denom_nce = (queries_u * d_sum.unsqueeze(0)).clamp(min=1e-8)
            f_nce     = term_contrib / denom_nce
            info.metrics.add(ScalarMetric(
                "f_nce", f_nce[mask].mean().item(), n_contributing.item()
            ))

            # Overlap metrics
            q_active  = queries_u > 0                                    # (N, V)
            q_nnz     = q_active.sum(dim=1).float().clamp(min=1)         # (N,)
            matched_pos = mask.sum(dim=1).float()                        # (N,)

            overlap_q_pos = matched_pos / q_nnz
            info.metrics.add(ScalarMetric(
                "overlap/q_pos", overlap_q_pos.mean().item(), N
            ))

            d_pos_nnz   = (d_pos > 0).sum(dim=1).float().clamp(min=1)   # (N,)
            overlap_d_pos = matched_pos / d_pos_nnz
            info.metrics.add(ScalarMetric(
                "overlap/d_pos", overlap_d_pos.mean().item(), N
            ))

            # In the pairwise case there is exactly 1 hard negative per query —
            # no need for the cross-product trick used in the batchwise path.
            matched_neg   = (q_active & (d_neg > 0)).sum(dim=1).float()  # (N,)
            overlap_q_neg = matched_neg / q_nnz
            info.metrics.add(ScalarMetric(
                "overlap/q_neg", overlap_q_neg.mean().item(), N
            ))

            d_neg_nnz   = (d_neg > 0).sum(dim=1).float().clamp(min=1)   # (N,)
            overlap_d_neg = matched_neg / d_neg_nnz
            info.metrics.add(ScalarMetric(
                "overlap/d_neg", overlap_d_neg.mean().item(), N
            ))

            hit_gap = overlap_q_pos - overlap_q_neg
            info.metrics.add(ScalarMetric(
                "overlap/hit_gap", hit_gap.mean().item(), N
            ))

            # Energy on matched tokens
            q_total_mass    = queries_u.sum(dim=1).clamp(min=1e-8)       # (N,)
            energy_q_match  = (queries_u * mask).sum(dim=1) / q_total_mass
            info.metrics.add(ScalarMetric(
                "energy/q_match", energy_q_match.mean().item(), N
            ))

            d_pos_total_mass = d_pos.sum(dim=1).clamp(min=1e-8)          # (N,)
            energy_d_match   = (d_pos * mask).sum(dim=1) / d_pos_total_mass
            info.metrics.add(ScalarMetric(
                "energy/d_match", energy_d_match.mean().item(), N
            ))

        elif rel is not None:
            # ------------------------------------------------------------------
            # Batchwise layout (BatchwiseTrainer / in-batch negatives).
            #
            # score_product is called, so the hook receives:
            #   queries   (B_q, V) — unique queries
            #   documents (B_d, V) — unique documents (each positive for one query)
            # current_relevances is a (B_q, B_d) matrix; argmax finds d+ index.
            # ------------------------------------------------------------------
            pos_idx = rel.argmax(dim=1).to(documents.device)  # (B,) moved to model device
            d_pos = documents[pos_idx]                         # (B, V) on model device

            # term_contrib[i, t] = q_i[t] * d+_i[t]
            # non-zero only where BOTH query i and its positive doc are active on t
            term_contrib = queries * d_pos                     # (B, V)

            # mask of (query, term) pairs that actually contribute to the score
            # all aggregations below are restricted to these positions only
            mask = term_contrib > 0                            # (B, V)
            n_contributing = mask.sum().clamp(min=1)           # scalar, used as count

            # --- F_base ---
            # f_base[i, t] = q_i[t]*d+_i[t] / Σ_j q_i[j]*d+_i[j]
            #   share of the positive dot-product carried by term t for query i
            total_score = term_contrib.sum(dim=1, keepdim=True).clamp(min=1e-8)
            f_base = term_contrib / total_score                # (B, V)

            # --- F_nce ---
            # f_nce[i, t] = q_i[t]*d+_i[t] / (q_i[t] * Σ_{all docs} d[t])
            #   how exclusively term t points to the positive vs. all in-batch docs
            d_sum = documents.sum(dim=0)                       # (V,)
            denom_nce = (queries * d_sum.unsqueeze(0)).clamp(min=1e-8)
            f_nce = term_contrib / denom_nce                   # (B, V)

            # Aggregate over contributing (query, term) pairs only — ignoring the
            # structural zeros that would arise from terms inactive in either the
            # query or the positive document.
            info.metrics.add(ScalarMetric(
                "f_base", f_base[mask].mean().item(), n_contributing.item()
            ))
            info.metrics.add(ScalarMetric(
                "f_nce", f_nce[mask].mean().item(), n_contributing.item()
            ))

            # ------------------------------------------------------------------
            # Structural overlap metrics — binary token co-activation
            #
            # overlap/q_pos  fraction of query terms landing in d+
            # overlap/d_pos  fraction of d+ touched by the query
            # overlap/q_neg  same as q_pos averaged over all in-batch negatives
            # overlap/d_neg  fraction of each negative touched by the query
            # overlap/hit_gap  q_pos − q_neg: structural discrimination margin
            # ------------------------------------------------------------------
            q_active = (queries > 0)                                     # (B, V)
            q_nnz = q_active.sum(dim=1).float().clamp(min=1)             # (B,)
            matched_pos = mask.sum(dim=1).float()                        # (B,) reuses mask

            overlap_q_pos = matched_pos / q_nnz
            info.metrics.add(ScalarMetric(
                "overlap/q_pos", overlap_q_pos.mean().item(), B_q
            ))

            d_pos_nnz = (d_pos > 0).sum(dim=1).float().clamp(min=1)     # (B,) reuses d_pos
            overlap_d_pos = matched_pos / d_pos_nnz
            info.metrics.add(ScalarMetric(
                "overlap/d_pos", overlap_d_pos.mean().item(), B_q
            ))

            # Negatives = all documents except each query's own positive.
            # cross_all[i, pos_idx[i]] == matched_pos[i] by definition, so the
            # positive contribution is subtracted to avoid double-counting.
            d_active_all = (documents > 0).float()                       # (B_d, V)
            cross_all = q_active.float() @ d_active_all.T                # (B, B_d)

            overlap_q_neg = (cross_all.sum(dim=1) - matched_pos) / ((B_d - 1) * q_nnz)
            info.metrics.add(ScalarMetric(
                "overlap/q_neg", overlap_q_neg.mean().item(), B_q
            ))

            d_nnz_all = d_active_all.sum(dim=1).clamp(min=1)             # (B_d,)
            coverage_all = cross_all / d_nnz_all.unsqueeze(0)            # (B, B_d)
            overlap_d_neg = (coverage_all.sum(dim=1) - overlap_d_pos) / (B_d - 1)
            info.metrics.add(ScalarMetric(
                "overlap/d_neg", overlap_d_neg.mean().item(), B_q
            ))

            hit_gap = overlap_q_pos - overlap_q_neg
            info.metrics.add(ScalarMetric(
                "overlap/hit_gap", hit_gap.mean().item(), B_q
            ))

            # ------------------------------------------------------------------
            # Energy on matched tokens
            # ------------------------------------------------------------------
            q_total_mass = queries.sum(dim=1).clamp(min=1e-8)            # (B,)
            energy_q_match = (queries * mask).sum(dim=1) / q_total_mass
            info.metrics.add(ScalarMetric(
                "energy/q_match", energy_q_match.mean().item(), B_q
            ))

            d_pos_total_mass = d_pos.sum(dim=1).clamp(min=1e-8)          # (B,) reuses d_pos
            energy_d_match = (d_pos * mask).sum(dim=1) / d_pos_total_mass
            info.metrics.add(ScalarMetric(
                "energy/d_match", energy_d_match.mean().item(), B_q
            ))


        if compute_flops:
            flops_q = (queries.mean(0) * queries.mean(0)).sum()
            flops_d = (documents.mean(0) * documents.mean(0)).sum()
            info.metrics.add(ScalarMetric("flops_q", flops_q.item(), 1))
            info.metrics.add(ScalarMetric("flops_d", flops_d.item(), 1))


class FlopsRegularizer(DualVectorListener):
    r"""The FLOPS regularizer computes

    .. math::

        FLOPS(q,d) = \lambda_q FLOPS(q) + \lambda_d FLOPS(d)

    where

    .. math::
        FLOPS(x) = \left( \frac{1}{d} \sum_{i=1}^d |x_i| \right)^2
    """

    lambda_q: Param[float]
    """Lambda for queries"""

    lambda_d: Param[float]
    """Lambda for documents"""

    def compute(x: torch.Tensor):
        """
        :param x: term vectors (batch size x vocabulary dimension)
        :returns: A couple (vocabulary dimension / FLOPS regularizations)
        """
        # Computes the mean for each term (weights are positive)
        y = x.mean(0)

        # Returns the sum of squared means
        return y, (y * y).sum()

    def __call__(self, info: TrainerContext, queries, documents):
        # queries and documents are length x dimension
        # Assumes that all weights are positive
        assert info.metrics is not None

        queries = queries.value
        documents = documents.value

        # q of shape (dimension), flops_q of shape (1)
        q, flops_q = FlopsRegularizer.compute(queries)
        d, flops_d = FlopsRegularizer.compute(documents)

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(Loss("flops", flops, 1.0))

        info.metrics.add(ScalarMetric("flops", flops.item(), 1))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), 1))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), 1))

        dual_representation_metrics(info, queries, documents)


class ScheduledFlopsRegularizer(FlopsRegularizer):
    """
    The FLOPS regularizer where the lamdba_q and lambda_d varie according to the
    steps. The lambda values goes quadratic before the
    ```lambda_warmup_steps```, and then remains constant
    """

    min_lambda_q: Param[float] = field(default=0, ignore_default=True)
    """Min value for the lambda_q before it increase"""

    min_lambda_d: Param[float] = field(default=0, ignore_default=True)
    """Min value for the lambda_d before it increase"""

    lambda_warmup_steps: Param[int] = field(default=0, ignore_default=True)
    """The warmup steps for the lambda"""

    def quadratic_ratio(self, step):
        if step > self.lambda_warmup_steps:
            return 1
        else:
            return (step / self.lambda_warmup_steps) ** 2

    def __post_init__(self):
        self.initial_lambda_q = self.lambda_q
        self.initial_lambda_d = self.lambda_d

    def __call__(self, info: TrainerContext, queries, documents):
        current_step = info.steps
        self.lambda_q = (
            self.initial_lambda_q - self.min_lambda_q
        ) * self.quadratic_ratio(current_step) + self.min_lambda_q
        self.lambda_d = (
            self.initial_lambda_d - self.min_lambda_d
        ) * self.quadratic_ratio(current_step) + self.min_lambda_d
        FlopsRegularizer.__call__(self, info, queries, documents)


class DFFlopsRegularizer(FlopsRegularizer):
    r"""FLOPS regularizer where the document side is weighted by a running
    estimate of per-term document frequency (DF).

    .. math::

        \ell_{DF\text{-}FLOPS} = \lambda_d \sum_{t \in V}
            \left( w_t \cdot \frac{1}{N} \sum_{i=1}^N r_{i,t} \right)^2
            + \lambda_q \cdot FLOPS(q)

    where :math:`w_t = \widehat{DF}_t / |C|` is maintained as an exponential
    moving average of the per-batch binary activation rate
    (:math:`(r_{i,t} > 0)`), approximating the corpus-level document
    frequency normalised by corpus size.  The query side is the standard
    (unweighted) FLOPS.
    """

    variant: Param[str] = "df-flops"
    """Constant string that distinguishes this regularizer from plain
    FlopsRegularizer in the experimaestro tag path.
    Always tag this param in the experiment so the job hash differs:
    ``ScheduledDFFlopsRegularizer.C(..., variant=tag("df-flops"))``."""

    ema_alpha: Param[float] = field(default=0.999, ignore_default=True)
    """EMA decay for the running DF estimate.
    Values close to 1 give a slow-moving, stable estimate; lower values
    track batch statistics more aggressively."""

    def __post_init__(self):
        # Running EMA of per-term DF (proportion of docs with term active).
        # Initialised lazily on the first call so we don't need vocab size here.
        self._ema_df: Optional[torch.Tensor] = None

    def _update_ema_df(self, documents: torch.Tensor) -> torch.Tensor:
        """Update the EMA and return current per-term weights w_t."""
        # Proportion of batch documents that activate each term — in [0, 1]
        batch_df = (documents > 0).float().mean(0).detach()  # (vocab,)
        if self._ema_df is None:
            self._ema_df = batch_df
        else:
            self._ema_df = (
                self.ema_alpha * self._ema_df.to(batch_df.device)
                + (1.0 - self.ema_alpha) * batch_df
            )
        return self._ema_df  # values in [0, 1]; activ = identity

    def __call__(self, info: TrainerContext, queries, documents):
        assert info.metrics is not None
        queries = queries.value
        documents = documents.value

        # Standard query FLOPS (unchanged)
        _, flops_q = FlopsRegularizer.compute(queries)

        # DF-weighted document FLOPS
        w = self._update_ema_df(documents)  # (vocab,)
        d_mean = documents.mean(0)          # (vocab,)
        flops_d = (w * d_mean).pow(2).sum()

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(Loss("flops", flops, 1.0))

        info.metrics.add(ScalarMetric("flops", flops.item(), 1))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), 1))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), 1))

        dual_representation_metrics(info, queries, documents)


class ScheduledDFFlopsRegularizer(DFFlopsRegularizer):
    """DF-FLOPS regularizer with quadratic lambda warmup.

    Identical schedule to :class:`ScheduledFlopsRegularizer`: both
    ``lambda_q`` and ``lambda_d`` ramp up quadratically from their
    ``min_lambda_*`` values over ``lambda_warmup_steps`` steps, then
    stay constant.
    """

    min_lambda_q: Param[float] = field(default=0, ignore_default=True)
    """Minimum lambda_q at the start of warmup."""

    min_lambda_d: Param[float] = field(default=0, ignore_default=True)
    """Minimum lambda_d at the start of warmup."""

    lambda_warmup_steps: Param[int] = field(default=0, ignore_default=True)
    """Steps over which lambdas ramp up quadratically."""

    def quadratic_ratio(self, step):
        if step > self.lambda_warmup_steps:
            return 1
        return (step / self.lambda_warmup_steps) ** 2

    def __post_init__(self):
        super().__post_init__()
        self.initial_lambda_q = self.lambda_q
        self.initial_lambda_d = self.lambda_d

    def __call__(self, info: TrainerContext, queries, documents):
        step = info.steps
        self.lambda_q = (
            self.initial_lambda_q - self.min_lambda_q
        ) * self.quadratic_ratio(step) + self.min_lambda_q
        self.lambda_d = (
            self.initial_lambda_d - self.min_lambda_d
        ) * self.quadratic_ratio(step) + self.min_lambda_d
        DFFlopsRegularizer.__call__(self, info, queries, documents)
