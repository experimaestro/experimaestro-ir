import itertools
from typing import Iterable, List, Optional
import torch
from xpmir.learning.batchers import Sliceable

from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer


class DualRepresentationScorer(LearnableScorer):
    """Neural scorer based on (at least a partially) independent representation
    of the document and the question.

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document tokens.
    """

    def forward(self, inputs: BaseRecords, info: Optional[TrainerContext] = None):
        # Forward to model
        enc_queries = self.encode_queries(
            [q.topic.get_text() for q in inputs.unique_queries]
        )
        enc_documents = self.encode_documents(
            [d.document.get_text() for d in inputs.unique_documents]
        )

        # Get the pairs
        pairs = inputs.pairs()
        q_ix, d_ix = pairs

        # TODO: Use a product query x document if possible

        return self.score_pairs(
            enc_queries[
                q_ix,
            ],
            enc_documents[
                d_ix,
            ],
            info,
        )

    def encode(self, texts: Iterable[str]):
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        raise NotImplementedError()

    def encode_documents(self, texts: Iterable[str]) -> Sliceable:
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        return self.encode(texts)

    def encode_queries(self, texts: Iterable[str]) -> Sliceable:
        """Encode a list of texts (document or query)

        The return value is model dependent, but should be sliceable

        By default, uses `merge`
        """
        return self.encode(texts)

    def merge_queries(self, list):
        """Merge query batches encoded with `encode_queries`

        By default, uses `merge`
        """
        return self.merge(list)

    def merge_documents(self, list):
        """Merge query batches encoded with `encode_documents`"""
        return self.merge(list)

    def merge(self, objects):
        """Merge objects

        - for tensors, uses torch.cat
        - for lists, concatenate all of them
        """
        assert isinstance(objects, List), "Merging can only be done with lists"

        if isinstance(objects[0], torch.Tensor):
            return torch.cat(objects)

        if isinstance(objects[0], List):
            return list(itertools.chain(objects))

        raise RuntimeError(f"Cannot deal with objects of type {type(list[0])}")

    def score_product(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        """Computes the score of all possible pairs of query and document

        Args:
            queries (Any): The encoded queries
            documents (Any): The encoded documents
            info (Optional[TrainerContext]): The training context (if learning)

        Returns:
            torch.Tensor:
                A tensor of dimension (N, P) where N is the number of queries
                and P the number of documents
        """
        raise NotImplementedError()

    def score_pairs(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        """Score the specified pairs of queries/documents.

        There are as many queries as documents. The exact type of
        queries and documents depends on the specific instance of the
        dual representation scorer.

        Args:
            queries (Any): The list of encoded queries
            documents (Any): The matching list of encoded documents
            info (Optional[TrainerContext]): _description_

        Returns:
            torch.Tensor:
                A tensor of dimension (N) where N is the number of documents/queries
        """
        raise NotImplementedError()
