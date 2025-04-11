from abc import abstractmethod
import itertools
from typing import Iterable, Union, List, Optional, TypeVar, Generic, Sequence
import torch
from datamaestro_text.data.ir import TextItem
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords, ProductRecords, TopicRecord, DocumentRecord
from xpmir.rankers import LearnableScorer

QueriesRep = TypeVar("QueriesRep", bound=Sequence)
DocsRep = TypeVar("DocsRep", bound=Sequence)


class DualRepresentationScorer(LearnableScorer, Generic[QueriesRep, DocsRep]):
    """Neural scorer based on (at least a partially) independent representation
    of the document and the question.

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document tokens.
    """

    def forward(self, inputs: BaseRecords, info: Optional[TrainerContext] = None):
        # Forward to model
        enc_queries = self.encode_queries(list(inputs.unique_queries))
        enc_documents = self.encode_documents(list(inputs.unique_documents))

        # Score product
        if isinstance(inputs, ProductRecords):
            return self.score_product(
                enc_queries,
                enc_documents,
                info,
            ).flatten()

        # Score pairs
        pairs = inputs.pairs()
        q_ix, d_ix = pairs
        return self.score_pairs(
            enc_queries[
                q_ix
            ],
            enc_documents[
                d_ix
            ],
            info,
        ).flatten()

    def encode(self, texts: Iterable[str]) -> Union[DocsRep, QueriesRep]:
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        raise NotImplementedError()

    def encode_documents(self, records: Iterable[DocumentRecord]) -> DocsRep:
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        return self.encode([record[TextItem].text for record in records])

    def encode_queries(self, records: Iterable[TopicRecord]) -> QueriesRep:
        """Encode a list of texts (document or query)

        The return value is model dependent, but should be sequence

        By default, uses `merge`
        """
        return self.encode([record[TextItem].text for record in records])

    def merge_queries(self, queries: QueriesRep):
        """Merge query batches encoded with `encode_queries`

        By default, uses `merge`
        """
        return self.merge(queries)

    def merge_documents(self, documents: DocsRep):
        """Merge query batches encoded with `encode_documents`"""
        return self.merge(documents)

    def merge(self, objects: Union[DocsRep, QueriesRep]):
        """Merge objects

        - for tensors, uses torch.cat
        - for lists, concatenate all of them
        """
        assert isinstance(
            objects, List
        ), f"Merging can only be done with lists, got {type(objects)}"

        # Just returns the only object to merge
        if len(objects) == 1:
            return objects[0]

        if isinstance(objects[0], torch.Tensor):
            return torch.cat(objects)

        if isinstance(objects[0], List):
            return list(itertools.chain(objects))

        from xpmir.text.encoders import TextsRepresentationOutput
        from xpmir.text.tokenizers import TokenizedTexts

        if isinstance(objects[0], TextsRepresentationOutput):

            def merge_mask(mask):
                min_batch_size = torch.min(
                    torch.tensor(list(map(lambda x: x.shape, mask))), dim=0
                )[0][0]
                batch_equalized = list(
                    itertools.chain(
                        *map(lambda x: x.t().split(min_batch_size, dim=1), mask)
                    )
                )

                pad = torch.nn.utils.rnn.pad_sequence(batch_equalized)

                return pad.reshape(pad.shape[0], -1).t()

            tokenized = list(map(lambda x: x.tokenized, objects))

            tokens = list(
                filter(lambda x: x is not None, map(lambda x: x.tokens, tokenized))
            )
            tokens = None if len(tokens) == 0 else list(itertools.chain(*tokens))

            ids = merge_mask(list(map(lambda x: x.ids, tokenized)))

            lens = list(itertools.chain(*map(lambda x: x.lens, tokenized)))

            mask = list(map(lambda x: x.mask, tokenized))
            mask = None if len(mask) == 0 else merge_mask(mask)

            token_type_ids = list(
                filter(
                    lambda x: x is not None, map(lambda x: x.token_type_ids, tokenized)
                )
            )
            token_type_ids = (
                None if len(token_type_ids) == 0 else torch.cat(token_type_ids)
            )

            return TextsRepresentationOutput(
                torch.cat(list(map(lambda x: x.value, objects))),
                TokenizedTexts(tokens, ids, lens, mask, token_type_ids),
            )
        raise RuntimeError(f"Cannot deal with objects of type {type(list[0])}")

    @abstractmethod
    def score_product(
        self,
        queries: QueriesRep,
        documents: DocsRep,
        info: Optional[TrainerContext] = None,
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
        ...

    @abstractmethod
    def score_pairs(
        self,
        queries: QueriesRep,
        documents: DocsRep,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        """Score the specified pairs of queries/documents.

        There are as many queries as documents. The exact type of
        queries and documents depends on the specific instance of the
        dual representation scorer.

        Args:
            queries (QueriesRep): The list of encoded queries
            documents (DocsRep): The matching list of encoded documents
            info (Optional[TrainerContext]): _description_

        Returns:
            torch.Tensor:
                A tensor of dimension (N) where N is the number of documents/queries
        """
        ...
