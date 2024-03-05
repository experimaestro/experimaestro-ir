import torch
from collections import namedtuple

from experimaestro import Param
from experimaestro.compat import cached_property
from torch import nn

from xpmir.letor.records import BaseRecords, TextItem
from xpmir.letor.trainers import TrainerContext
from xpmir.neural.generative import (
    ConditionalGenerator,
    StepwiseGenerator,
    BeamSearchGenerationOptions,
)
from xpmir.neural.generative.hf import FullSequenceGenerationOutput
from xpmir.learning import ModuleInitOptions
from xpmir.rankers import AbstractModuleScorer
from xpmir.utils.utils import easylog
from xpmir.neural.generative.referential import DepthUpdatable

logger = easylog()

# dataclass for inference, compose of doc and qry
PairwiseTuple = namedtuple("PairwiseTuple", ["doc", "qry"])


class GenerativeRetrievalScorer(AbstractModuleScorer, DepthUpdatable):
    """The abstract class for the generative retrieval scorer"""

    id_generator: Param[ConditionalGenerator]
    """The id generator"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.id_generator.initialize(options)

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the scorer"
            )
        else:
            self.current_max_depth = self.max_depth


# --- For inference
class NaiveGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A naive scorer which will be used for the inference of the generative
    retrieval model, and this scorer is not learnable"""

    early_finish_punishment: Param[int] = 1
    """A early finish punishment hyperparameter, trying to make model score less
    if the id list is too short. Default value to 1 means no punishment"""

    def recursive(
        self,
        decoder_input_tokens,  # shape [bs]
        unfinished_sequences,
        depth,  # a value counting from 1
        stepwise_generators: PairwiseTuple,
    ):
        assert depth <= self.current_max_depth
        # pass get the probas
        logits = PairwiseTuple(
            *[g.step(decoder_input_tokens) for g in stepwise_generators]
        )
        log_proba = PairwiseTuple(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )

        # take the maximum indices
        next_tokens = torch.max(torch.exp(log_proba.qry), dim=-1).indices

        batch_range = torch.arange(len(next_tokens))
        log_proba_next = PairwiseTuple(
            *(x[batch_range, next_tokens] for x in log_proba)
        )

        # mask them! For the sequence already finished, replaced by the
        # multiplier 1(or some other multiplier to punish the early finish)
        log_proba_next.doc[unfinished_sequences == 0] = self.early_finish_punishment
        log_proba_next.qry[unfinished_sequences == 0] = self.early_finish_punishment

        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return torch.exp(log_proba_next.qry)

        return self.recursive(
            next_tokens,
            new_unfinished_sequences,
            depth + 1,
            stepwise_generators,
        ) * torch.exp(log_proba_next.doc)

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        self.id_generator.eval()
        queries_text = [pdr[TextItem].text for pdr in inputs.topics]
        documents_text = [ndr[TextItem].text for ndr in inputs.documents]

        bs = len(queries_text)

        stepwise_generator = PairwiseTuple(
            *[self.id_generator.stepwise_iterator() for _ in range(2)]
        )

        stepwise_generator.qry.init(queries_text)
        stepwise_generator.doc.init(documents_text)

        # initialization
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        return self.recursive(
            decoder_input_tokens, unfinished_sequences, 1, stepwise_generator
        )


# --- For inference
class RandomBasedGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A scorer based on the probability that a document is better than a random
    document given a query, and this scorer is not learnable"""

    @cached_property
    def random_distribution(self):  # shape [max_depth, decoder_outdim+1]
        assert self.id_generator.eos_token_id == self.id_generator.decoder_outdim
        decoder_outdim = self.id_generator.decoder_outdim
        alphas = torch.tensor(
            [
                sum(decoder_outdim**i for i in range(j + 1))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.id_generator.device)
        alphas = (1 / alphas).unsqueeze(-1)
        return torch.log(
            torch.cat(
                (((1 - alphas) / decoder_outdim).expand(-1, decoder_outdim), alphas), -1
            )
        )

    def recursive(
        self,
        decoder_input_tokens,  # shape [bs]
        unfinished_sequences,
        depth,  # a value counting from 1
        stepwise_generators: PairwiseTuple,
    ):
        assert depth <= self.current_max_depth
        # pass get the probas
        logits = PairwiseTuple(
            *[g.step(decoder_input_tokens) for g in stepwise_generators]
        )
        log_proba = PairwiseTuple(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )

        log_proba_randdoc = self.random_distribution[depth - 1]
        # -- the exact term of the scorer
        exact_term = torch.sum(
            torch.exp(log_proba.qry + log_proba.doc)
            * (1 - torch.exp(log_proba_randdoc)),
            dim=-1,
        )

        # take the sampled indices
        next_tokens = torch.multinomial(
            torch.exp(log_proba.qry), num_samples=1
        ).squeeze(1)

        batch_range = torch.arange(len(next_tokens))
        log_proba_next = PairwiseTuple(
            *(x[batch_range, next_tokens] for x in log_proba)
        )

        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return unfinished_sequences * exact_term

        # generate the proba for the random doc, according to whether the
        # sampled one is the eos
        log_proba_next_randdoc = torch.where(
            next_tokens == self.id_generator.eos_token_id,
            log_proba_randdoc[self.id_generator.eos_token_id],
            log_proba_randdoc[0],
        )

        return unfinished_sequences * (
            self.recursive(
                next_tokens,
                new_unfinished_sequences,
                depth + 1,
                stepwise_generators,
            )
            * torch.exp(log_proba_next.doc + log_proba_next_randdoc)
            + exact_term
        )

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        self.id_generator.eval()
        queries_text = [pdr[TextItem].text for pdr in inputs.topics]
        documents_text = [ndr[TextItem].text for ndr in inputs.documents]

        bs = len(queries_text)

        stepwise_generator = PairwiseTuple(
            *[self.id_generator.stepwise_iterator() for _ in range(2)]
        )

        stepwise_generator.qry.init(queries_text)
        stepwise_generator.doc.init(documents_text)

        # initialization
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        return self.recursive(
            decoder_input_tokens, unfinished_sequences, 1, stepwise_generator
        )


class BeamSearchRandomBasedGenerativeRetrievalScorer(
    RandomBasedGenerativeRetrievalScorer
):
    """A scorer based on the probability that a document is better than a random
    document given a query, but we choose plusieur paths for the query"""

    num_beams: Param[int] = 16

    def recursive(
        self,
        depth,  # a value counting from 1
        stepwise_generator: StepwiseGenerator,
        query_generate_output: FullSequenceGenerationOutput,
    ):
        assert depth <= self.current_max_depth
        # if all generated queries ends before reaching the max depth, we need
        # to adapt it.
        current_max_depth = query_generate_output.sequences.shape[-1] - 1

        decoder_input_tokens = (
            None if depth == 1 else query_generate_output.sequences[:, depth - 1]
        )
        # pass get the probas
        document_logits = stepwise_generator.step(decoder_input_tokens)
        document_log_proba = nn.functional.log_softmax(
            document_logits, dim=-1
        )  # [bs*num_beam, vocab_size]
        log_proba_randdoc = self.random_distribution[depth - 1]

        # -- the exact term of the scorer
        exact_term = torch.sum(
            torch.exp(query_generate_output.all_scores[depth - 1] + document_log_proba)
            * (1 - torch.exp(log_proba_randdoc)),
            dim=-1,
        )  # shape [bs*num_beam]

        # get the tokens passed for the next layers --> predefined by query
        next_tokens = query_generate_output.sequences[:, depth]  # shape [bs*num_beam]
        # if encounter eos, need to replace by another token or we will
        # encounter index out of range error, we can use whatever token id cause
        # it will be masked later.
        # Here it use the 0 as an example
        next_tokens = torch.where(
            next_tokens == self.id_generator.pad_token_id, 0, next_tokens
        )
        batch_range = torch.arange(len(next_tokens))
        log_proba_next_doc = document_log_proba[batch_range, next_tokens]

        log_proba_next_randdoc = torch.where(
            next_tokens == self.id_generator.eos_token_id,
            log_proba_randdoc[self.id_generator.eos_token_id],
            log_proba_randdoc[0],
        )
        unfinished_sequences = query_generate_output.output_mask[:, depth - 1]
        # here no need to set the new_unfinished_squeneces cause if all the
        # sequences are finished, there are no tokens at that depth, which means
        # reach the current_max_depth
        if depth == current_max_depth:
            return unfinished_sequences * exact_term  # shape [bs*num_beam]

        return unfinished_sequences * (
            self.recursive(
                depth + 1,
                stepwise_generator,
                query_generate_output,
            )
            * torch.exp(log_proba_next_doc + log_proba_next_randdoc)
            + exact_term
        )  # shape [bs*num_beam]

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        self.id_generator.eval()
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        query_generate_options = BeamSearchGenerationOptions(
            max_new_tokens=self.current_max_depth,
            num_return_sequences=self.num_beams,
            num_beams=self.num_beams,
        )
        query_generate_output: FullSequenceGenerationOutput = (
            self.id_generator.generate(queries_text, query_generate_options)
        )
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]
        documents_text_repeat = [
            d for d in documents_text for _ in range(self.num_beams)
        ]

        document_stepwise_generator = self.id_generator.stepwise_iterator()
        document_stepwise_generator.init(documents_text_repeat)

        recursive_output = self.recursive(
            1, document_stepwise_generator, query_generate_output
        )  # shape [bs, num_beams]

        # sequence_scores_sum = torch.sum(
        #     torch.exp(query_generate_output.sequence_scores).reshape(
        #         -1, self.num_beams
        #     ),
        #     dim=-1,
        # )
        # info.add_metric(
        #     ScalarMetric(
        #         "Query total portion", float(sequence_scores_sum), len(queries_text)
        #     )
        # )
        return torch.sum(
            (
                recursive_output * torch.exp(query_generate_output.sequence_scores)
            ).reshape(-1, self.num_beams),
            -1,
        )


class StateBasedBeamSearchGenerativeRetrievalScorer(
    RandomBasedGenerativeRetrievalScorer
):
    """A scorer based on the probability that a document is better than a random
    document given a query, but we choose plusieur paths for the query,
    we save the state to increase the efficiency"""

    num_beams: Param[int] = 16

    def query_output_slice(
        self,
        original_query_output: FullSequenceGenerationOutput,
        beam_slice: slice,
    ):
        # slice the original query output to get the Query output which contains
        # only the desired part
        new_all_scores = [
            all_score[beam_slice] for all_score in original_query_output.all_scores
        ]
        return FullSequenceGenerationOutput(
            sequences=original_query_output.sequences[beam_slice],
            output_mask=original_query_output.output_mask[beam_slice],
            transition_scores=original_query_output.transition_scores[beam_slice],
            all_scores=new_all_scores,
            sequence_scores=original_query_output.sequence_scores[beam_slice],
        )

    def sort_query_output(
        self,
        original_query_output: FullSequenceGenerationOutput,
    ):
        # first get the sequence and convert to the (dim+2)-base number
        base = self.id_generator.decoder_outdim
        current_max_depth = original_query_output.sequences.shape[-1] - 1
        base_v = torch.tensor(
            [base ** (i - 1) for i in range(current_max_depth, 0, -1)]
        ).to(self.id_generator.device)
        integer_value = torch.sum(original_query_output.sequences[:, 1:] * base_v, -1)
        _, indice = torch.sort(integer_value)
        new_all_scores = [
            all_score[indice] for all_score in original_query_output.all_scores
        ]
        return FullSequenceGenerationOutput(
            sequences=original_query_output.sequences[indice],
            output_mask=original_query_output.output_mask[indice],
            transition_scores=original_query_output.transition_scores[indice],
            all_scores=new_all_scores,
            sequence_scores=original_query_output.sequence_scores[indice],
        )

    def recursive(
        self,
        bs,
        depth,  # a value counting from 1
        stepwise_generator: StepwiseGenerator,
        query_generate_output: FullSequenceGenerationOutput,
    ):
        assert depth <= self.current_max_depth
        # if all generated queries ends before reaching the max depth, we need
        # to adapt it.
        current_max_depth = query_generate_output.sequences.shape[-1] - 1
        decoder_input_tokens = (
            None
            if depth == 1
            else query_generate_output.sequences[0, depth - 1].expand(bs)
        )

        # pass get the probas
        document_logits = stepwise_generator.step(decoder_input_tokens)
        document_log_proba = nn.functional.log_softmax(
            document_logits, dim=-1
        )  # [bs, vocab_size]
        log_proba_randdoc = self.random_distribution[depth - 1]  # shape [vocab_size]

        # -- the exact term of the scorer
        if depth == current_max_depth:
            # at the max_depth
            unfinished_sequences = query_generate_output.output_mask[
                :, depth - 1
            ]  # shape [final_layer_beam]
            exact_term = torch.sum(
                torch.exp(query_generate_output.all_scores[depth - 1].unsqueeze(0))
                * torch.exp(document_log_proba.unsqueeze(1))
                * (1 - torch.exp(log_proba_randdoc)),
                dim=-1,
            )  # shape [bs*last_layer_beam]
            return unfinished_sequences * exact_term  # shape [bs*final_layer_beam]
        else:
            # if it is not at the last layer, the exact_term should be all same
            # for all beams as they have the same prefix
            exact_term = torch.sum(
                torch.exp(
                    query_generate_output.all_scores[depth - 1][0] + document_log_proba
                )
                * (1 - torch.exp(log_proba_randdoc)),
                dim=-1,
            )  # shape [bs]

        # return the count for each child,
        # use it to generate the indice for the next layer
        # next_tokens.shape [nb_child], count_shape [nb_child]
        next_tokens, counts = torch.unique(
            query_generate_output.sequences[:, depth], return_counts=True
        )
        nb_child = counts.shape[0]

        tmp = [0] + torch.cumsum(counts, 0).tolist()
        query_beam_slices = [
            slice(tmp[i], tmp[i + 1]) for i in range(len(tmp) - 1)
        ]  # len(nb_child)
        cumulate = []
        if nb_child > 1:
            state = stepwise_generator.state()
        for i in range(nb_child):
            next_token = next_tokens[i]  # int
            unfinished_sequences = (
                0 if self.id_generator.pad_token_id == next_token else 1
            )
            # for the same next_token, the log_proba_next_doc should be the same
            log_proba_next_doc = document_log_proba[:, next_token]  # shape [bs]
            log_proba_next_randdoc = (
                log_proba_randdoc[self.id_generator.eos_token_id]
                if next_token == self.id_generator.eos_token_id
                else log_proba_randdoc[0]
            )  # int
            child_res = unfinished_sequences * (
                self.recursive(
                    bs,
                    depth + 1,
                    stepwise_generator,
                    self.query_output_slice(
                        query_generate_output, query_beam_slices[i]
                    ),
                ).transpose(0, 1)
                * torch.exp(log_proba_next_doc + log_proba_next_randdoc)
                + exact_term
            ).transpose(
                0, 1
            )  # shape [bs*next_layer_beam]
            cumulate.append(child_res)
            if nb_child > 1:
                # restore the state for the next child
                stepwise_generator.load_state(state)

        return torch.cat(cumulate, dim=-1)  # shape [bs, current_layer_beam]

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        self.id_generator.eval()
        queries_text = [pdr[TextItem].text for pdr in inputs.unique_topics]
        assert len(queries_text) == 1, "Should contains only one query for one batch"
        query_generate_options = BeamSearchGenerationOptions(
            max_new_tokens=self.current_max_depth,
            num_return_sequences=self.num_beams,
            num_beams=self.num_beams,
        )
        query_generate_output: FullSequenceGenerationOutput = (
            self.id_generator.generate(queries_text, query_generate_options)
        )

        # sort the query_generate_output for easy recursive
        # e.g [[3,4,5,7], [3,4,5,2], [3,2,6,4], [3,4,2,3]] will be rerank to
        # [[3,2,6,4],[3,4,2,3],[3,4,5,2],[3,4,5,7]]
        sorted_query_generate_output: FullSequenceGenerationOutput = (
            self.sort_query_output(query_generate_output)
        )

        documents_text = [ndr[TextItem].text for ndr in inputs.documents]

        document_stepwise_generator = self.id_generator.stepwise_iterator()
        document_stepwise_generator.init(documents_text)
        bs = len(documents_text)
        recursive_output = self.recursive(
            bs, 1, document_stepwise_generator, sorted_query_generate_output
        )

        return torch.sum(
            recursive_output
            * torch.exp(sorted_query_generate_output.sequence_scores).reshape(
                -1, self.num_beams
            ),
            -1,
        )
