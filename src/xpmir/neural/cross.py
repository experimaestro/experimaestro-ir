import torch
from typing import Tuple
from experimaestro import Param
from datamaestro_text.data.ir import TextItem
from xpmir.distributed import DistributableModel
from xpmir.learning.batchers import Batcher
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import (
    BaseRecords,
    PairwiseRecords,
)
from xpmir.rankers import LearnableScorer
from xpmir.text import TokenizerOptions
from xpmir.text.encoders import TextEncoderBase, TripletTextEncoder
from xpmir.rankers import (
    DuoLearnableScorer,
    DuoTwoStageRetriever,
    Retriever,
)
from xpmir.utils.utils import easylog

logger = easylog()


class CrossScorer(LearnableScorer, DistributableModel):
    """Query-Document Representation Classifier

    Based on a query-document representation representation (e.g. BERT [CLS] token).
    AKA Cross-Encoder
    """

    max_length: Param[int]
    """Maximum length (in tokens) for the query-document string"""

    encoder: Param[TextEncoderBase[Tuple[str, str], torch.Tensor]]
    """an encoder for encoding the concatenated query-document tokens which
    doesn't contains the final linear layer"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.encoder.initialize(options)
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)
        self.tokenizer_options = TokenizerOptions(max_length=self.max_length)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        pairs = self.encoder(
            [
                (tr[TextItem].text, dr[TextItem].text)
                for tr, dr in zip(inputs.topics, inputs.documents)
            ],
            # options=self.tokenizer_options,
        )  # shape (batch_size * dimension)
        return self.classifier(pairs.value).squeeze(1)

    def distribute_models(self, update):
        self.encoder = update(self.encoder)


class DuoCrossScorer(DuoLearnableScorer, DistributableModel):
    """Preference based classifier

    This scorer can be used to train a DuoBERT-type model.
    """

    encoder: Param[TripletTextEncoder]
    """The encoder to use for the Duobert model"""

    def __validate__(self):
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.encoder.initialize(options)
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
