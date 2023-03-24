from attrs import define
from experimaestro import experiment, setmeta, copyconfig
from functools import cached_property
import logging

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro.launcherfinder import find_launcher
from xpmir.datasets.adapters import (
    RandomFold,
    RetrieverBasedCollection,
    MemoryTopicStore,
)
from xpmir.evaluation import Evaluations, EvaluationsCollection
import xpmir.interfaces.anserini as anserini
from xpmir.letor.batchers import PowerAdaptativeBatcher

from xpmir.letor.samplers import Sampler
from xpmir.letor.samplers import TripletBasedSampler, PairwiseInBatchNegativesSampler
from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.utils.utils import find_java_home
from xpmir.letor.distillation.samplers import (
    DistillationPairwiseSampler,
    PairwiseHydrator,
)
from xpmir.models import AutoModel
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.distributed import DistributedHook
from xpmir.rankers.full import FullRetriever

from ..pipelines.msmarco import MSMarcoV1Configuration, MSMarcoV1Experiment


logging.basicConfig(level=logging.INFO)


@define(kw_only=True)
class SPLADEMSMarcoV1Configuration(MSMarcoV1Configuration):
    # Nothing new for now
    pass


class SPLADEMSMarcoV1Experiment(MSMarcoV1Experiment):
    """Base class for SPLADE experiments based on MS-Marco v1"""

    cfg: SPLADEMSMarcoV1Configuration

    ds_val: RetrieverBasedCollection
    """Validation dataset, different from RandomFold because we want harder
    negatives"""

    tests: EvaluationsCollection
    """The collections on which the models are evaluated"""

    @cached_property
    def splade_sampler(self) -> Sampler:
        """Retrurn different types of trainer based on different configuration"""
        # define the trainer based on different dataset
        if self.cfg.learner.dataset == "":
            train_triples = prepare_dataset(
                "irds.msmarco-passage.train.docpairs"
            )  # pair for pairwise learner

            triplesid = ShuffledTrainingTripletsLines(
                seed=123,
                data=train_triples,
            ).submit()

            # generator a batchwise sampler which is an Iterator of ProductRecords()
            # TODO: could replace the self.index with the self.document or but
            # for now it may destroy the id
            train_sampler = TripletBasedSampler(
                source=triplesid, index=self.index
            )  # the pairwise sampler from the dataset.
            return PairwiseInBatchNegativesSampler(
                sampler=train_sampler
            )  # generating the batchwise from the pairwise

        elif self.cfg.learner.dataset == "bert_hard_negative":
            # hard negatives trained by distillation with cross-encoder
            # Improving Efficient Neural Ranking Models with Cross-Architecture
            # Knowledge Distillation, (Sebastian Hofstätter, Sophia Althammer,
            # Michael Schröder, Mete Sertkan, Allan Hanbury), 2020
            # In the form of Tuple[Query, Tuple[Document, Document]] without text
            train_triples_distil = prepare_dataset(
                "com.github.sebastian-hofstaetter."
                + "neural-ranking-kd.msmarco.ensemble.teacher"
            )

            # All the query text
            train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

            # Combine the training triplets with the document and queries texts
            distillation_samples = PairwiseHydrator(
                samples=train_triples_distil,
                documentstore=self.documents,
                querystore=MemoryTopicStore(topics=train_topics),
            )

            # Generate a sampler from the samples
            return DistillationPairwiseSampler(samples=distillation_samples)

    def __init__(self, xp: experiment, cfg: SPLADEMSMarcoV1Configuration):
        super().__init__(xp, cfg)

        # launcher for the index
        self.cpu_launcher_index = find_launcher(cfg.indexation.requirements)
        self.gpu_launcher_index = find_launcher(cfg.indexation.training_requirements)

        # Sets the working directory and the name of the xp
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())

        # define the test set
        self.tests = EvaluationsCollection(
            msmarco_dev=Evaluations(self.devsmall, self.measures),
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), self.measures
            ),
        )
        # Index for msmarcos
        # TODO: need to update to ```anserini.indexbuilder()```
        self.index = anserini.IndexCollection(
            documents=self.documents, storeContents=True
        ).submit()

        # Build a dev. collection for full-ranking (validation) "Efficiently
        # Teaching an Effective Dense Retriever with Balanced Topic Aware
        # Sampling"
        tasb = AutoModel.load_from_hf_hub(
            "xpmir/tas-balanced"
        )  # create a scorer from huggingface

        # task to train the tas_balanced encoder for the document list and
        # generate an index for retrieval
        tasb_index = IndexBackedFaiss(
            indexspec=cfg.tas_balance_retriever.indexspec,
            device=self.device,
            normalize=False,
            documents=self.documents,
            sampler=RandomDocumentSampler(
                documents=self.documents,
                max_count=cfg.tas_balance_retriever.faiss_max_traindocs,
            ),  # Just use a fraction of the dataset for training
            encoder=DenseDocumentEncoder(scorer=tasb),
            batchsize=2048,
            batcher=PowerAdaptativeBatcher(),
            hooks=[
                setmeta(
                    DistributedHook(models=[tasb.encoder, tasb.query_encoder]), True
                )
            ],
        ).submit(launcher=self.gpu_launcher_index)

        # A retriever if tas-balanced. We use the index of the faiss.
        # Used it to create the validation dataset.
        tasb_retriever = FaissRetriever(
            index=tasb_index,
            topk=cfg.tas_balance_retriever.retTopK,
            encoder=DenseQueryEncoder(scorer=tasb),
        )

        # building the validation dataset. Based on the existing dataset and the
        # top retrieved doc from tas-balanced and bm25
        self.ds_val = RetrieverBasedCollection(
            dataset=RandomFold(
                dataset=self.dev,
                seed=123,
                fold=0,
                sizes=[cfg.learner.validation_size],
                exclude=self.devsmall.topics,
            ).submit(),
            retrievers=[
                tasb_retriever,
                anserini.AnseriniRetriever(
                    k=cfg.tas_balance_retriever.retTopK,
                    index=self.index,
                    model=self.basemodel,
                ),
            ],
        ).submit(launcher=self.gpu_launcher_index)

        # compute the baseline performance on the test dataset.
        # Bm25
        bm25_retriever = anserini.AnseriniRetriever(
            k=cfg.base_retriever.topK, index=self.index, model=self.basemodel
        )
        self.tests.evaluate_retriever(
            copyconfig(bm25_retriever).tag("model", "bm25"), self.cpu_launcher_index
        )

        # tas-balance
        self.tests.evaluate_retriever(
            copyconfig(tasb_retriever).tag("model", "tasb"), self.gpu_launcher_index
        )

        # Base retrievers for validation
        # It retrieve all the document of the collection with score 0
        self.base_retriever_full = FullRetriever(documents=self.ds_val.documents)
