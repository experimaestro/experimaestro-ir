from errno import EROFS
import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.letor.context import TrainerContext
from xpmir.letor.records import BaseRecords, PairwiseRecord, PairwiseRecords, Query, Document
from xpmir.neural import TorchLearnableScorer
from xpmir.text.encoders import DualTextEncoder, TripletTextEncoder
from xpmir.rankers import DuoLearnableScorer, ScoredDocument
from typing import List, Optional

class CrossScorer(TorchLearnableScorer):
    """Query-Document Representation Classifier

    Based on a query-document representation representation (e.g. BERT [CLS] token).
    AKA Cross-Encoder

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder; if null, uses the document encoder
    """

    encoder: Param[DualTextEncoder]

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        pairs = self.encoder(
            [(q.text, d.text) for q, d in zip(inputs.queries, inputs.documents)]
        )
        return self.classifier(pairs).squeeze(1)

class DuoCrossScorer(DuoLearnableScorer):
    """Query-document-document Representation classifier based on Bert
    The encoder usually refer to the encoder of type DualDuoBertTransformerEncoder()
    """

    encoder: Param[TripletTextEncoder]

    def  __validate__(self):
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def forward(self, inputs: PairwiseRecords, info: TrainerContext = None):
        """Encode the query-document-document
        """
        triplets = self.encoder(
            [(q.text, d_1.text, d_2.text) 
            for q, d_1, d_2 in zip(inputs.unique_queries, inputs.positives, inputs.negatives)]
        )
        return self.classifier(triplets).squeeze(1)

    

    def rsv(self, query: str, documents: List[ScoredDocument], aggregation: str = 'sum') -> List[ScoredDocument]: 
        """Do the forward pass and then aggregate the scores for a document
        """
        query: Param[str]
        """The query to retrieve by the model"""

        documents: Param[List[ScoredDocument]]
        """The top K_2 documents which preselected by the monobert"""

        aggregation: Optional[str] = 'sum'
        """The type of the aggregation function to use"""

        def calculate_aggregation_score(input: torch.Tensor, k: int, aggregation: str) -> torch.Tensor:
            prob_factory = input.reshape(k, -1)
            if aggregation == 'sum':
                return torch.sum(prob_factory, dim = 1)
            if aggregation == 'max':
                return torch.max(prob_factory, dim = 1)
            if aggregation == 'min': 
                return torch.min(prob_factory, dim = 1)
            if aggregation == 'binary':
                return torch.sum(prob_factory > 0.5, dim = 1)
            raise RuntimeError(f"{aggregation} not supported!")

        inputs = PairwiseRecords()
        for doc in documents:
            assert doc.content is not None

        qry = Query(None, query)
        k = len(documents)
        for i in range(k):
            document_i = Document(documents[i].docid, documents[i].content, documents[i].score)
            for j in range(k):
                if i != j:
                    document_j = Document(documents[j].docid, documents[j].content, documents[j].score)
                    inputs.add(PairwiseRecord(qry, document_i, document_j))

        with torch.no_grad():
            scores = self(inputs, None).cpu() # len((k)*(k-1))
            scores_aggregate = calculate_aggregation_score(scores, k, aggregation)

            # add the scored documents to the list
            scoredDocuments = []
            for i in range(k):
                scoredDocuments.append(ScoredDocument(documents[i].docid, float(scores_aggregate[i])))

        return scoredDocuments