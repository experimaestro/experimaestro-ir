import torch
from experimaestro import Param
from xpmir.distributed import DistributableModel
from xpmir.learning.batchers import Batcher
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import (
    BaseRecords,
    PairwiseRecords,
)
from xpmir.rankers import LearnableScorer
from xpmir.text.encoders import DualTextEncoder, TripletTextEncoder
from xpmir.rankers import (
    DuoLearnableScorer,
    DuoTwoStageRetriever,
    Retriever,
)


class CrossScorer(LearnableScorer, DistributableModel):
    """Query-Document Representation Classifier

    Based on a query-document representation representation (e.g. BERT [CLS] token).
    AKA Cross-Encoder

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder; if null, uses the document encoder
    """

    encoder: Param[DualTextEncoder]
    """an encoder for encoding the concatenated query-document tokens which
    doesn't contains the final linear layer"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        pairs = self.encoder(
            [
                (tr.topic.get_text(), dr.document.get_text())
                for tr, dr in zip(inputs.topics, inputs.documents)
            ]
        )  # shape (batch_size * dimension)
        return self.classifier(pairs).squeeze(1)

    def distribute_models(self, update):
        self.encoder.model = update(self.encoder.model)


class DuoCrossScorer(DuoLearnableScorer, DistributableModel):
    """Query-document-document Representation classifier based on Bert
    The encoder usually refer to the encoder of type DualDuoBertTransformerEncoder()
    """

    encoder: Param[TripletTextEncoder]
    """The encoder to use for the Duobert model"""

    def __validate__(self):
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def forward(self, inputs: PairwiseRecords, info: TrainerContext = None):
        """Encode the query-document-document"""
        triplets = self.encoder(
            [
                (q.text, d_1.text, d_2.text)
                for q, d_1, d_2 in zip(
                    inputs.unique_queries, inputs.positives, inputs.negatives
                )
            ]
        )
        return self.classifier(triplets).squeeze(1)

    def distribute_models(self, update):
        self.encoder.model = update(self.encoder.model)

    def getRetriever(
        self,
        retriever: "Retriever",
        batch_size: int,
        batcher: Batcher = Batcher(),
        top_k=None,
        device=None,
    ):
        """The given the base_retriever and return a two stage retriever
        specialized for Duobert"""
        return DuoTwoStageRetriever(
            retriever=retriever,
            scorer=self,
            batchsize=batch_size,
            batcher=batcher,
            device=device,
            top_k=top_k,
        )
