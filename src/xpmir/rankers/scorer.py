# This package contains all rankers
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TYPE_CHECKING,
    TypedDict,
)
from typing_extensions import ReadOnly
from xpmir.text import TokenizedTexts
import torch
import torch.nn as nn
from lightning_fabric import Fabric
from experimaestro import Param, Config, Meta, field, tqdm
from datamaestro_ir.data import (
    Documents,
    IDTextRecord,
    SimpleTextItem,
)
import os
import torch.distributed as dist
from xpm_torch import Module, Random
from xpm_torch.utils.utils import Initializable
from xpm_torch.utils.logging import EasyLogger
from xpm_torch.datasets import IndexedDataset, ShardedIterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from xpm_torch.learner import TrainerContext
from xpm_torch.losses import ModuleOutputType
from xpmir.letor.records import (
    BaseItems,
    PairwiseItem,
    PairwiseItems,
    PointwiseItem,
    PointwiseItems,
    ProductItems,
)
from datamaestro_ir.data.base import ScoredDocument
from .retriever import Retriever

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from xpmir.evaluation import RetrieverFactory


class Scorer(Config, Initializable, EasyLogger, ABC):
    """Query-document scorer

    A model able to give a score to a list of documents given a query
    """

    _initialized = False

    outputType: ModuleOutputType = ModuleOutputType.REAL
    """Determines the type of output scalar (log probability, probability, logit) """

    doc: Meta[str] = ""
    """Paper description or title (used in HF Hub README)"""

    bibtex: Meta[str] = ""
    """BibTeX citation (used in HF Hub README)"""

    def __initialize__(self):
        """Initialize the scorer"""
        pass

    def rsv(
        self,
        topic: Union[str, IDTextRecord],
        documents: Union[List[ScoredDocument], ScoredDocument, str, List[str]],
    ) -> List[ScoredDocument]:
        """Compute the Retrieval Status Value (RSV) for a query and a set of documents.

        This method is the primary entry point for scoring a set of documents
        against a single query. It handles input normalization and delegates
        to the :meth:`compute` method.

        Note:
            For large-scale evaluation involving multiple queries, using
            :meth:`Retriever.retrieve_all` via a :class:`TwoStageRetriever`
            is preferred as it allows for cross-query batching on GPUs.
        """
        # Convert into document records
        if isinstance(documents, str):
            documents = [ScoredDocument({"text_item": SimpleTextItem(documents)}, None)]
        elif isinstance(documents[0], str):
            documents = [
                ScoredDocument({"text_item": SimpleTextItem(scored_document)}, None)
                for scored_document in documents
            ]

        # Convert into topic record
        if isinstance(topic, str):
            topic = {"text_item": SimpleTextItem(topic)}

        return self.compute(topic, documents)

    @abstractmethod
    def compute(
        self, topic: IDTextRecord, documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        """Score all documents with respect to a single topic.

        This method should be implemented by subclasses to provide the actual
        scoring logic. It is query-atomic (processes one query at a time).
        """
        ...

    def getRetriever(
        self,
        retriever: "Retriever",
        batch_size: int,
        top_k=None,
        device=None,
    ):
        """Returns a two stage re-ranker from this retriever and a scorer

        :param device: Device for the ranker or None if no change should be made

        :param batch_size: The number of documents in each batch

        :param top_k: Number of documents to re-rank (or None for all)
        """
        return TwoStageRetriever.C(
            retriever=retriever,
            scorer=self,
            batchsize=batch_size,
            top_k=top_k if top_k else None,
        )


def scorer_retriever(
    documents: Documents,
    *,
    retrievers: "RetrieverFactory",
    scorer: Scorer,
    key: str = None,
    **kwargs,
):
    """Helper function that returns a two stage retriever. This is useful
    when used with partial (when the scorer is not known).

    :param documents: The document collection
    :param retrievers: A retriever factory
    :param scorer: The scorer
    :return: A retriever, calling the :meth:scorer.getRetriever
    """
    assert retrievers is not None, "The retrievers have not been given"
    assert scorer is not None, "The scorer has not been given"
    return scorer.getRetriever(retrievers(documents, key=key), **kwargs)


class RandomScorer(Scorer):
    """A random scorer"""

    random: Param[Random]
    """The random number generator"""

    def compute(
        self, record: IDTextRecord, scored_documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        result = []
        random = self.random.state
        for scored_document in scored_documents:
            result.append(ScoredDocument(scored_document.document, random.random()))
        return result


class AbstractModuleScorerCall(Protocol):
    def __call__(
        self,
        inputs: "BaseItems",
        *,
        tokenized: Optional[TokenizedTexts] = None,
    ): ...


class AbstractModuleScorer(Scorer, Module):
    """Base class for all torch-based Modules implementing the `xpmir.rankers.Scorer`.

    While :meth:`compute` (inherited from :class:`Scorer`) processes documents
    for a single query, :class:`AbstractModuleScorer` also supports cross-query
    batching when called directly through its `forward` method (aliased as `__call__`).

    When used in a :class:`TwoStageRetriever` with a `batchsize > 0`, the retriever
    will use the :class:`PointwiseItems` batching to maximize GPU utilization across
    multiple queries.
    """

    # Ensures basic operations are redirected to torch.nn.Module methods
    __call__: AbstractModuleScorerCall = nn.Module.__call__
    train = nn.Module.train

    def __init__(self):
        logger.info(f"Initializing {self.__class__.__name__}")
        nn.Module.__init__(self)
        super().__init__()
        self._initialized = False

    def __initialize__(self):
        """Initialize a learnable scorer (structure only)"""
        return self

    def get_forward_methods(self) -> list:
        """Returns the list of forward methods for this scorer. By default, it is just `forward`, but it can be extended to support multiple forward methods (e.g. for different scoring strategies)"""
        return ["rsv"]

    def compute(
        self, topic: IDTextRecord, scored_documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        """Atomic scoring for a single query using ProductItems.

        This implementation leverages the :meth:`forward` method by wrapping
        the single query and its documents into a :class:`ProductItems` object.
        """
        # Prepare the inputs and call the model
        inputs = ProductItems()
        inputs.add_topics(topic)

        inputs.add_documents(*[sd.document for sd in scored_documents])

        with torch.no_grad():
            scores = self(inputs, None).cpu().float().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i, sd in enumerate(scored_documents):
            scoredDocuments.append(
                ScoredDocument(
                    sd.document,
                    float(scores[i].item()),
                )
            )

        return scoredDocuments


class DuoLearnableScorer(AbstractModuleScorer):
    """Base class for models that can score a triplet (query, document 1, document 2)"""

    def forward(self, inputs: "PairwiseItems", info: Optional[TrainerContext]):
        """Returns scores for pairs of documents (given a query)"""
        raise NotImplementedError(f"abstract __call__ in {self.__class__}")


class AbstractTwoStageRetriever(Retriever):
    """Abstract class for all two stage retrievers (i.e. scorers and duo-scorers)"""

    retriever: Param[Retriever]
    """The base retriever"""

    scorer: Param[Scorer]
    """The scorer used to re-rank the documents"""

    top_k: Param[Optional[int]]
    """The number of returned documents (if None, returns all the documents)"""

    batchsize: Meta[int] = field(default=0, ignore_default=True)
    """The batch size for the re-ranker"""

    def initialize(self):
        self.retriever.initialize()
        self.scorer.initialize()


class RerankingInputs(TypedDict):
    """Inputs for re-ranking"""

    records: ReadOnly[PointwiseItems]
    """The pointwise records"""

    batch: ReadOnly[List[PointwiseItem]]
    """The original batch of items"""

    tokenized_records: ReadOnly[Optional[TokenizedTexts]]
    """The tokenized records (if any)"""


def reranking_collate(
    batch: List[PointwiseItem],
) -> RerankingInputs:
    """Collate PointwiseItems into a PointwiseItems batch."""
    batch_items = PointwiseItems()
    for item in batch:
        batch_items.add(item)
    return RerankingInputs(records=batch_items, batch=batch, tokenized_records=None)


class ReRankingDataset(ShardedIterableDataset):
    """A dataset that yields PointwiseItem records for re-ranking"""

    def __init__(self, queries: Dict[str, IDTextRecord], retriever: Retriever):
        super().__init__()
        self.queries = list(queries.items())
        self.retriever = retriever

    def iter_shard(self, shard_id: int, num_shards: int):
        import os

        rank = os.environ.get(
            "SLURM_PROCID", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        )
        logger.debug(
            "ReRankingDataset (rank %s): shard %d/%d starting (total queries: %d)",
            rank,
            shard_id,
            num_shards,
            len(self.queries),
        )
        for i in range(shard_id, len(self.queries), num_shards):
            qid, query = self.queries[i]
            # Pull first-stage results on-the-fly to avoid materialising everything
            scored_docs = self.retriever.retrieve(query)
            for sd in scored_docs:
                yield PointwiseItem(query, sd.document, sd.score)


class TwoStageRetriever(AbstractTwoStageRetriever):
    """Use on retriever to select the top-K documents which are the re-ranked
    given a scorer.

    Multi-GPU support:
        When set up with a :class:`lightning.Fabric` instance, :meth:`retrieve_all`
        shards the re-ranking task across GPUs and gathers the results. It uses
        efficient cross-query batching to maximize GPU throughput.
    """

    def retrieve(self, record: IDTextRecord):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(record)

        # Score all documents
        _scoredDocuments = self.scorer.rsv(record, scoredDocuments)

        _scoredDocuments.sort(reverse=True)
        return _scoredDocuments[: (self.top_k or len(_scoredDocuments))]

    def build_reranking_dataloader(self, queries: Dict[str, IDTextRecord]):
        """Builds a dataloader for re-ranking all documents for a set of queries
        Allows for efficient two-stage retrieval with cross-query batching on GPU when the scorer supports it (i.e. is an AbstractModuleScorer and batchsize > 0).
        """
        # We don't materialise everything, but iterate on the fly
        dataset = ReRankingDataset(queries, self.retriever)

        # get underlying module if wrapped (e.g. Fabric)
        scorer = self.scorer.module if hasattr(self.scorer, "module") else self.scorer

        if hasattr(scorer, "get_tokenizer_fn"):
            tokenization_fn = scorer.get_tokenizer_fn()

            def collate_fn(batch: List[PointwiseItem]) -> RerankingInputs:
                inputs = reranking_collate(batch)
                inputs["tokenized_records"] = tokenization_fn(inputs["records"])
                return inputs

        else:
            collate_fn = reranking_collate

        dataloader = StatefulDataLoader(
            dataset,
            batch_size=self.batchsize,
            num_workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 4),
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn,
        )

        fabric: Fabric = getattr(self, "fabric", None)
        if fabric:
            dataloader = fabric.setup_dataloaders(dataloader)

        return dataloader

    @torch.no_grad()
    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        """Retrieves documents for all queries in an efficient two - stage fashion:
        - populate a `PointWiseItem` dataset with the documents from first stage
        - reranks them on the fly with the scorer with given batch size
        - if self.batchsize is 0, scores all documents from the same query at once (will cause OOM large top_k first stages)
        """
        # Scorer in evaluation mode
        self.scorer.eval()

        if self.batchsize == 0:
            # Fallback to per-query retrieval if no batchsize
            return super().retrieve_all(queries)

        # get underlying module if wrapped (e.g. Fabric)
        scorer = self.scorer.module if hasattr(self.scorer, "module") else self.scorer
        scorer_type = type(scorer)

        fabric: Fabric = getattr(self, "fabric", None)
        dataloader = self.build_reranking_dataloader(queries)

        # Process in batches
        scored_results = {qid: [] for qid in queries}
        seen_qids = set()

        disable_tqdm = fabric is not None and not fabric.is_global_zero

        # Calculate local total for the progress bar to reach 100% on rank 0
        total_to_process = len(queries)
        if fabric and fabric.world_size > 1:
            # Number of queries this specific rank (0) will handle
            total_to_process = len(
                range(fabric.global_rank, len(queries), fabric.world_size)
            )

        pbar = (
            tqdm(
                total=total_to_process,
                desc="Re-ranking",
                unit="query",
            )
            if not disable_tqdm
            else None
        )

        if issubclass(scorer_type, AbstractModuleScorer):
            logger.info(
                f"Re-Ranking with '{scorer_type.__name__}' using batch size {self.batchsize}..."
            )
        else:
            logger.info(
                f"Re-Ranking with '{scorer_type.__name__}' with rsv (one-by-one)..."
            )
        for inputs in dataloader:
            batch_items = inputs["records"]
            batch = inputs["batch"]
            tokenized_records = inputs.get("tokenized_records")

            if issubclass(scorer_type, AbstractModuleScorer):
                # Use scorer.forward if it's an AbstractModuleScorer to batch across queries

                scores = (
                    self.scorer(batch_items, tokenized=tokenized_records)
                    .cpu()
                    .float()
                    .numpy()
                )

                for score, item in zip(scores, batch):
                    qid = item.topic["id"]
                    if qid not in seen_qids:
                        seen_qids.add(qid)
                        if pbar is not None:
                            pbar.update(1)
                    scored_results[qid].append(
                        ScoredDocument(item.document, float(score.item()))
                    )
            else:
                # Fallback: group by query and use rsv (score one by one)
                by_query = {}
                for item in batch:
                    qid = item.topic["id"]
                    if qid not in seen_qids:
                        seen_qids.add(qid)
                        if pbar is not None:
                            pbar.update(1)
                    by_query.setdefault(qid, []).append(
                        ScoredDocument(item.document, item.relevance)
                    )

                for qid, docs in by_query.items():
                    scored_results[qid].extend(self.scorer.rsv(queries[qid], docs))

        if pbar is not None:
            pbar.close()

        if fabric and fabric.world_size > 1:
            # Gather results from all GPUs
            # We use a list of tuples and filter empty results to reduce
            # communication overhead. Using torch.distributed directly for
            # objects is more robust than fabric.all_gather which is tensor-oriented.
            local_results = [
                (qid, docs) for qid, docs in scored_results.items() if docs
            ]
            gathered_data = [None] * fabric.world_size
            dist.all_gather_object(gathered_data, local_results)

            if fabric.is_global_zero:
                # Merge them back into one master dictionary
                final_results = {qid: [] for qid in queries}
                for rank_results in gathered_data:
                    for qid, docs in rank_results:
                        final_results[qid].extend(docs)
                scored_results = final_results
            else:
                return {}

        # Sort and truncate
        for qid in scored_results:
            scored_results[qid].sort(reverse=True)
            if self.top_k:
                scored_results[qid] = scored_results[qid][: self.top_k]

        return scored_results


class DuoTwoStageRetriever(AbstractTwoStageRetriever):
    """The two stage retriever for pairwise scorers.

    For pairwise scorer, we need to aggregate the pairwise scores in some
    way.
    """

    def retrieve(self, query: IDTextRecord):
        """call the _retrieve function by using the batcher and do an
        aggregation of all the scores
        """
        # get the documents from the retriever
        scoredDocuments_previous = self.retriever.retrieve(query)

        # Scorer in evaluation mode
        self.scorer.eval()

        # Generator for pairs to avoid materialising everything
        def iter_pairs():
            for i in range(len(scoredDocuments_previous)):
                for j in range(len(scoredDocuments_previous)):
                    if i != j:
                        yield (scoredDocuments_previous[i], scoredDocuments_previous[j])

        _scores_pairs = []  # the scores for each pair of documents
        if self.batchsize > 0:
            # Duo-reranking often involves a small number of docs (N=50, 100),
            # but N^2 can still be large (10k). We use a list for now as
            # IndexedDataset needs a sequence, but the materialisation is limited
            # to ONE query at a time.
            pairs = list(iter_pairs())
            dataset = IndexedDataset(pairs)
            dataloader = StatefulDataLoader(
                dataset, batch_size=self.batchsize, collate_fn=lambda x: x
            )
            for batch in dataloader:
                _scores_pairs.extend(self.rsv(query, batch))
        else:
            pairs = list(iter_pairs())
            _scores_pairs = self.rsv(query, pairs)

        # Use the sum aggregation strategy
        _scores_pairs = torch.Tensor(_scores_pairs).reshape(
            len(scoredDocuments_previous), -1
        )
        _scores_per_document = torch.sum(
            _scores_pairs, dim=1
        )  # scores for each document.

        # construct the ScoredDocument object from the score we just get.
        scoredDocuments = []
        for i in range(len(scoredDocuments_previous)):
            scoredDocuments.append(
                ScoredDocument(
                    scoredDocuments_previous[i], float(_scores_per_document[i])
                )
            )
        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: (self.top_k or len(scoredDocuments))]

    def rsv(
        self,
        record: IDTextRecord,
        documents: List[Tuple[ScoredDocument, ScoredDocument]],
    ) -> List[float]:
        """Given the query and documents in tuple
        return the score for each triplets
        """
        inputs = PairwiseItems()
        for doc1, doc2 in documents:
            inputs.add(PairwiseItem(record, doc1, doc2))

        with torch.no_grad():
            scores = self.scorer(inputs, None).cpu().float()  # shape (batchsizes)
            return scores.tolist()
