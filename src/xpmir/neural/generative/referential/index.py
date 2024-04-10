from typing import List, Optional, Tuple, Any
from experimaestro import Param, Meta, tqdm
from datamaestro_text.data.ir import Documents, DocumentRecord, TextItem
import torch

from xpmir.learning.batchers import Batcher
from xpmir.learning import ModuleInitMode
from xpmir.letor import Device
from xpmir.letor.records import TopicRecord
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.neural.generative import ConditionalGenerator
from xpmir.neural.generative.referential import BeamSearchGenerationOptions
from xpmir.text.encoders import (
    TextEncoderBase,
    InputType,
    EncoderOutput,
    RepresentationOutput,
)


class ReferentialEncoder(TextEncoderBase[InputType, EncoderOutput]):
    id_generator: Param[ConditionalGenerator]
    """The generator to encoder the document text into the vector"""

    max_depth: Param[int]
    """The maximum depth of the model"""

    num_sequences: Param[int]
    """The number number of sequences generated for each document"""

    def __initialize__(self, options):
        super().__initialize__(options)
        self.id_generator.initialize(options)
        # list all the possible sequences
        self.all_sequences = []
        # build this dictionary recursively
        self.mapping_builder([], 1)
        self.docid_to_indice = {
            sequence: indice for indice, sequence in enumerate(self.all_sequences)
        }  # e.g. {"1\t6\t7": 23}

    @property
    def dimension(self) -> int:
        return len(self.all_sequences)

    def to(self, *args, **kwargs):
        self.id_generator.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.id_generator.train(*args, **kwargs)
        super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.id_generator.eval(*args, **kwargs)
        super().eval(*args, **kwargs)

    def mapping_builder(
        self,
        prefix: List[int],
        depth: int,
    ):
        prefix_str = "\t".join(map(str, prefix))
        if depth == self.max_depth + 1:
            self.all_sequences.append(prefix_str)
            return

        for i in range(self.id_generator.decoder_outdim):
            new_prefix = prefix.copy()
            new_prefix.append(i)
            self.mapping_builder(new_prefix, depth + 1)

        self.all_sequences.append(prefix_str)
        return

    def forward(self, texts: List[InputType]) -> EncoderOutput:
        """encoder the texts to a vector"""
        bs = len(texts)
        generate_options = BeamSearchGenerationOptions(
            max_new_tokens=self.max_depth,
            num_return_sequences=self.num_sequences,
            num_beams=self.num_sequences,
        )
        document_output = self.id_generator.generate(texts, generate_options)
        # transform the output matrix to
        raw_sequences = document_output.sequences[:, 1:]
        raw_sequences_list = document_output.sequences[:, 1:].tolist()
        mask_list = torch.sum(
            torch.logical_and(
                raw_sequences != self.id_generator.eos_token_id,
                raw_sequences != self.id_generator.pad_token_id,
            ).int(),
            dim=-1,
        ).tolist()

        # get the keys of the generated sequences
        sequences_str = [
            "\t".join(map(str, raw_sequence[:num_keep]))
            for raw_sequence, num_keep in zip(raw_sequences_list, mask_list)
        ]
        # get the correponding indice
        indices = (
            torch.tensor([self.docid_to_indice[key] for key in sequences_str])
            .reshape(bs, self.num_sequences)
            .to(self._dummy_params.device)
        )
        values = document_output.sequence_scores.reshape(-1, self.num_sequences)
        results = torch.zeros(bs, self.dimension).to(self._dummy_params.device)
        return RepresentationOutput(value=results.scatter_add_(1, indices, values))


class ReferentialValidationRescorer(Retriever):
    """This retriever rescore the given documents in the validation set"""

    encoder: Param[ReferentialEncoder]
    """The encoder to encode the query and document to sparse vectors"""

    documents: Param[Documents]
    """The validation corpus"""

    batchsize: Param[int] = 128
    batcher: Meta[Batcher] = Batcher()
    device: Meta[Optional[Device]] = None

    def initialize(self):
        self.encode_batcher = self.batcher.initialize(self.batchsize)
        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options())
        if self.device is not None:
            self.encoder.to(self.device.value)

    def encode_query(
        self, queries: List[Tuple[str, TopicRecord]], encoded: List[Any], pbar
    ):
        """Encode the queries"""
        encoded.append(
            self.encoder([record[TextItem].text for _, record in queries]).value
        )
        pbar.update(len(queries))
        return encoded

    def score(
        self,
        documents: List[DocumentRecord],
        queries: torch.Tensor,
        scored_documents: List[List[ScoredDocument]],
        pbar,
    ):
        # Encode documents
        encoded = self.encoder([record[TextItem].text for record in documents])

        new_scores = [[] for _ in documents]
        for ix in range(queries.shape[0]):
            # Get a range of query records
            query = queries[ix : (ix + 1)]

            # Returns a query x document matrix
            scores = (
                query.to(self.device.value) @ encoded.value.T
            )  # shape [1, document_bs]

            # Adds up to the lists
            scores = scores.flatten().detach()  # shape [document_bs]
            for ix, (document, score) in enumerate(zip(documents, scores)):
                new_scores[ix].append(ScoredDocument(document, float(score)))
                pbar.update(1)

        # Add each result to the full document list
        scored_documents.extend(new_scores)

    def retrieve(self, record: DocumentRecord) -> List[ScoredDocument]:
        return self.retrieve_all({"_": record})["_"]

    def retrieve_all(
        self, queries: torch.Dict[str, TopicRecord]
    ) -> torch.Dict[str, List[ScoredDocument]]:
        self.encoder.eval()
        all_queries = list(queries.items())  # bs
        with torch.no_grad():
            # Encode all queries
            # each time the batcher will just encode a batchsize of queries
            # and then concat them together
            with tqdm(total=len(all_queries), desc="Encoding queries") as pbar:
                enc_queries = self.encode_batcher.reduce(
                    all_queries, self.encode_query, [], pbar
                )
            enc_queries = torch.cat(enc_queries)  # tensor shape (bs, dim)

            scored_documents: List[List[ScoredDocument]] = []
            with tqdm(
                total=len(all_queries) * self.documents.documentcount,
                desc="Scoring documents",
            ) as pbar:
                self.encode_batcher.process(
                    self.documents, self.score, enc_queries, scored_documents, pbar
                )

        qids = [qid for qid, _ in all_queries]
        return {qid: [sd[ix] for sd in scored_documents] for ix, qid in enumerate(qids)}
