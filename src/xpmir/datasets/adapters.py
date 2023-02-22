from typing import Iterable, Iterator, List, Optional, Set, Union
from pathlib import Path
from experimaestro import (
    Param,
    Config,
    Task,
    tqdm,
    cache,
    pathgenerator,
    Annotated,
    Meta,
)
from experimaestro.compat import cached_property
from datamaestro_text.data.ir import (
    Adhoc,
    AdhocAssessments,
    AdhocDocument,
    AdhocDocumentStore,
    AdhocDocuments,
    AdhocTopics,
)

from datamaestro_text.data.ir.trec import TrecAdhocAssessments
from datamaestro_text.data.ir.csv import AdhocTopics as CSVAdhocTopics
from xpmir.rankers import Retriever
from xpmir.utils.utils import easylog

logger = easylog()


class AdhocTopicFold(AdhocTopics):
    """ID-based topic selection"""

    ids: Param[List[str]]
    """A set of the ids for the topics where we select from"""

    topics: Param[AdhocTopics]
    """The collection of the topics"""

    def iter(self):
        ids = set(self.ids)
        for topic in self.topics.iter():
            if topic.qid in ids:
                yield topic


class AdhocAssessmentFold(AdhocAssessments):
    ids: Param[List[str]]
    """A set of the ids for the assessments where we select from"""

    qrels: Param[AdhocAssessments]
    """The collection of the assessments"""

    @cache("assessements.qrels")
    def trecpath(self, path):
        ids = set(self.ids)
        if not path.is_file():
            with path.open("wt") as fp:
                for qrels in self.iter():
                    if qrels.qid in ids:
                        for qrel in qrels.assessments:
                            fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")

        return path

    def iter(self):
        ids = set(self.ids)
        for qrels in self.qrels.iter():
            if qrels.qid in ids:
                yield qrels


def fold(ids: Iterable[str], dataset: Adhoc):
    """Returns a fold of a dataset, given topic ids"""
    ids = sorted(list(ids))
    topics = AdhocTopicFold(topics=dataset.topics, ids=ids)
    qrels = AdhocAssessmentFold(assessments=dataset.assessments, ids=ids)
    return Adhoc(topics=topics, assessments=qrels, documents=dataset.documents)


class ConcatFold(Task):
    """
    Concatenation of several datasets to get a full dataset.
    """

    datasets: Param[List[Adhoc]]
    """The list of Adhoc datasets to concatenate"""

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Generated assessments file"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Generated topics file"""

    def config(self) -> Adhoc:
        dataset_document_id = set(dataset.document.id for dataset in self.datasets)
        assert (
            len(dataset_document_id) == 1
        ), "At the moment only one set of documents supported."
        return Adhoc(
            id="",  # No need to have a more specific id since it is generated
            topics=CSVAdhocTopics(id="", path=self.topics),
            assessments=TrecAdhocAssessments(id="", path=self.assessments),
            documents=self.datasets[0].documents,
        )

    def execute(self):
        topics = []
        # concat the topics
        for dataset in self.datasets:
            topics.extend([topic for topic in dataset.topics.iter()])

        # Write topics and assessments
        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic.qid)
                slash_t = "\t"
                fp.write(f"""{topic.qid}\t{topic.text.replace(slash_t, ' ')}\n""")

        with self.assessments.open("wt") as fp:
            for dataset in self.datasets:
                for qrels in dataset.assessments.iter():
                    if qrels.qid in ids:
                        for qrel in qrels.assessments:
                            fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")


class RandomFold(Task):
    """Extracts a random subset of topics from a dataset"""

    seed: Param[int]
    """Random seed used to compute the fold"""

    sizes: Param[List[float]]
    """Number of topics of each fold (or percentage if sums to 1)"""

    dataset: Param[Adhoc]
    """The Adhoc dataset from which a fold is extracted"""

    fold: Param[int]
    """Which fold should be taken"""

    exclude: Param[Optional[AdhocTopics]]
    """Exclude some topics from the random fold"""

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Generated assessments file"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Generated topics file"""

    def __validate__(self):
        assert self.fold < len(self.sizes)

    @staticmethod
    def folds(
        seed: int,
        sizes: List[float],
        dataset: Param[Adhoc],
        exclude: Param[Optional[AdhocTopics]] = None,
        submit=True,
    ):
        """Creates folds

        Parameters:

        - submit: if true (default), submits the fold tasks to experimaestro
        """

        folds = []
        for ix in range(len(sizes)):
            fold = RandomFold(
                seed=seed, sizes=sizes, dataset=dataset, exclude=exclude, fold=ix
            )
            if submit:
                fold = fold.submit()
            folds.append(fold)

        return folds

    def config(self) -> Adhoc:
        return Adhoc(
            id="",  # No need to have a more specific id since it is generated
            topics=CSVAdhocTopics(id="", path=self.topics),
            assessments=TrecAdhocAssessments(id="", path=self.assessments),
            documents=self.dataset.documents,
        )

    def execute(self):
        import numpy as np

        # Get topics
        badids = (
            set(topic.qid for topic in self.exclude.iter()) if self.exclude else set()
        )
        topics = [
            topic for topic in self.dataset.topics.iter() if topic.qid not in badids
        ]
        random = np.random.RandomState(self.seed)
        random.shuffle(topics)

        # Get the fold
        sizes = np.array([0.0] + self.sizes)
        s = sizes.sum()
        if abs(s - 1) < 1e-6:
            sizes = np.round(len(topics) * sizes)
            sizes = np.round(len(topics) * sizes / sizes.sum())

        assert sizes[self.fold + 1] > 0

        indices = sizes.cumsum().astype(int)
        topics = topics[indices[self.fold] : indices[self.fold + 1]]

        # Write topics and assessments
        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic.qid)
                fp.write(f"""{topic.qid}\t{topic.text}\n""")

        with self.assessments.open("wt") as fp:
            for qrels in self.dataset.assessments.iter():
                if qrels.qid in ids:
                    for qrel in qrels.assessments:
                        fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")


class AdhocDocumentSubset(AdhocDocuments):
    """ID-based topic selection"""

    base: Param[AdhocDocumentStore]
    """The full document store"""

    docids_path: Meta[Path]
    """Path to the file containing the document IDs"""

    def __len__(self):
        return len(self.docids)

    @property
    def documentcount(self):
        return len(self.docids)

    def __getitem__(self, slice: Union[int, slice]):
        docids = self.docids[slice]
        if isinstance(docids, List):
            return AdhocDocumentSubsetSlice(
                self, docids, range(len(self.docids))[slice]
            )
        return AdhocDocument(docids, self.base.document_text(docids), slice)

    @cached_property
    def docids(self) -> List[str]:
        # Read document IDs
        with self.docids_path.open("rt") as fp:
            return [line.strip() for line in fp]

    def iter_ids(self):
        yield from self.docids

    def iter(self) -> Iterator[AdhocDocument]:
        for docid in self.iter_ids():
            content = self.base.document_text(docid)
            yield AdhocDocument(docid, content)


class AdhocDocumentSubsetSlice:
    def __init__(
        self, subset: AdhocDocumentSubset, docids: List[str], internal_ids: List[int]
    ):
        self.subset = subset
        self.docids = docids
        self.internal_ids = internal_ids

    def __iter__(self):
        for internal_docid, docid in zip(self.internal_ids, self.docids):
            yield AdhocDocument(
                docid,
                self.subset.base.document_text(docid),
                internal_docid=internal_docid,
            )

    def __len__(self):
        return len(self.docids)

    def __getitem__(self, ix):
        docid = self.docids[ix]
        internal_docid = self.internal_ids[ix]
        return AdhocDocument(
            docid, self.subset.base.document_text(docid), internal_docid=internal_docid
        )


class RetrieverBasedCollection(Task):
    """Buils a subset of documents based on the output of a set of retrievers
    and on relevance assessment.
    First get all the document based on the assessment then add the retrieved ones.
    """

    relevance_threshold: Param[float] = 0
    """Relevance threshold"""

    dataset: Param[Adhoc]
    """A dataset"""

    retrievers: Param[List[Retriever]]
    """Rankers"""

    keepRelevant: Param[bool] = True
    """Keep documents judged relevant"""

    keepNotRelevant: Param[bool] = False
    """Keep documents judged not relevant"""

    docids_path: Annotated[Path, pathgenerator("docids.txt")]
    """The file containing the document identifiers of the collection"""

    def __validate__(self):
        assert len(self.retrievers) > 0, "At least one retriever should be given"

    def config(self) -> Adhoc:
        return Adhoc(
            id="",  # No need to have a more specific id since it is generated
            topics=self.dataset.topics,
            assessments=self.dataset.assessments,
            documents=AdhocDocumentSubset(
                id="", base=self.dataset.documents, docids_path=self.docids_path
            ),
        )

    def execute(self):
        for retriever in self.retrievers:
            retriever.initialize()

        # Selected document IDs
        docids: Set[str] = set()

        topics = {t.qid: t for t in self.dataset.assessments.iter()}

        # Retrieve all documents
        for topic in tqdm(
            self.dataset.topics.iter(), total=self.dataset.topics.count()
        ):
            qrels = topics.get(topic.qid)
            if qrels is None:
                logger.warning(
                    "Skipping topic %s [%s], (no assessment)", topic.qid, topic.text
                )
                continue

            # Add (not) relevant documents
            if self.keepRelevant:
                docids.update(
                    a.docno
                    for a in qrels.assessments
                    if a.rel > self.relevance_threshold
                )

            if self.keepNotRelevant:
                docids.update(
                    a.docno
                    for a in qrels.assessments
                    if a.rel <= self.relevance_threshold
                )

            # Retrieve and add
            # already defined the numbers to retrieve inside the retriever, so
            # don't need to worry about the threshold here
            for retriever in self.retrievers:
                docids.update(sd.docid for sd in retriever.retrieve(topic.text))

        # Write the document IDs
        with self.docids_path.open("wt") as fp:
            fp.writelines(f"{docid}\n" for docid in docids)


class TextStore(Config):
    """Associates an ID with a text"""

    def __getitem__(self, key: str) -> str:
        raise NotImplementedError()


class MemoryTopicStore(TextStore):
    """View a set of topics as a (in memory) text store"""

    topics: Param[AdhocTopics]
    """The collection of the topics to build the store"""

    @cached_property
    def store(self):
        return {topic.qid: topic.text for topic in self.topics.iter()}

    def __getitem__(self, key: str) -> str:
        return self.store[key]
