import dataclasses
from typing import List

import torch
from experimaestro import Config, Param
from experimaestro.compat import cached_property
from torch import LongTensor, FloatTensor
from transformers import LogitsProcessor, LogitsProcessorList

from xpmir.learning.context import StepTrainingHook
from xpmir.letor.trainers import TrainerContext
from xpmir.neural.generative import (
    ConditionalGenerator,
    GenerateOptions,
    StepwiseGenerator,
    BeamSearchGenerationOptions,
)
from xpmir.neural.generative.hf import FullSequenceGenerationOutput
from xpmir.learning import ModuleInitOptions
from xpmir.utils.utils import easylog

logger = easylog()


class DepthUpdatable(Config):
    """Abstract class of the objects which could update their depth"""

    max_depth: Param[int] = 5
    """The max number of the steps we need to consider, counting from 1"""

    current_max_depth: int
    """The max_depth for the current learning stage in the progressive training
    stage"""

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the loss"
            )
        else:
            self.current_max_depth = self.max_depth

    def __post_init__(self):
        # if no update
        logger.info(f"If no further updates, the max_depth we use is {self.max_depth}")
        self.current_max_depth = self.max_depth


class GenRetDepthUpdateHook(StepTrainingHook):
    """Update the depth of the training instance(loss, scorer, etc) procedure"""

    objects: Param[List[DepthUpdatable]]
    """The objects to update the depth during the learning procedure"""

    update_interval: Param[int] = 200
    """The interval to update the learning depth"""

    def before(self, state: TrainerContext):
        # start with depth 1
        if state.steps % (self.update_interval * state.steps_per_epoch) == 1:
            current_depth = (state.epoch - 1) // self.update_interval + 1
            for object in self.objects:
                object.update_depth(current_depth)


class DepthBasedSequenceBiasLogitsProcessor(LogitsProcessor):
    """Only support the bias term of length 1, and the bias is only added at the eos"""

    def __init__(self, sequence_bias: torch.tensor) -> None:
        super().__init__()
        self.sequence_bias = sequence_bias  # shape [bs*num_beam, decoder_dim+1]

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        # We don't need to renormalize the logits here, we can just set
        # renormalize_logits=True during the generation
        current_depth = input_ids.shape[1]  # counting from 1, 2, ..
        return scores + self.sequence_bias[current_depth - 1]


# The stepwise generator for the model with additional bias
class GeneratorBiasStepwiseGenerator(StepwiseGenerator):
    def __init__(
        self,
        id_generator: ConditionalGenerator,
        stepwise_iterator: StepwiseGenerator,
    ):
        super().__init__()
        # The identifier to use to generate the next step's token
        self.id_generator = id_generator
        self.stepwise_iterator = stepwise_iterator

    def init(self, texts: List[str]):
        self.stepwise_iterator.init(texts)
        self.current_depth = 0

    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        logits = self.stepwise_iterator.step(token_ids)
        bs = logits.shape[0]
        logits = logits + self.id_generator.bias_terms[self.current_depth].expand(
            bs, -1
        )
        self.current_depth += 1
        return logits


# The model with addtional bias
class GeneratorBiasAdapter(ConditionalGenerator):
    max_depth: Param[int] = 5
    """The max_depth of the generator"""

    vanilla_generator: Param[ConditionalGenerator]
    """The original generator, for the moment a T5ConditionalGenerator"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.vanilla_generator.initialize(options)
        self.decoder_outdim = self.vanilla_generator.decoder_outdim
        self.eos_token_id = self.vanilla_generator.eos_token_id
        self.pad_token_id = self.vanilla_generator.pad_token_id
        self.decoder_start_token_id = self.vanilla_generator.decoder_start_token_id

    def stepwise_iterator(self) -> StepwiseGenerator:
        return GeneratorBiasStepwiseGenerator(
            self, self.vanilla_generator.stepwise_iterator()
        )

    @property
    def device(self):
        return self.vanilla_generator.device

    @cached_property
    def bias_terms(self):
        assert (
            self.vanilla_generator.eos_token_id == self.vanilla_generator.decoder_outdim
        )
        decoder_dim = self.vanilla_generator.decoder_outdim
        alphas = torch.tensor(
            [
                sum(decoder_dim**i for i in range(j))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.device)
        alphas = torch.log((1 / alphas)).unsqueeze(-1)
        return torch.cat(
            (torch.zeros(alphas.shape[0], decoder_dim).to(self.device), alphas), -1
        )

    def generate(self, inputs: List[str], options: GenerateOptions = None):
        assert options.return_dict_in_generate, "Must return dict in this case"
        assert options.output_scores, "Must return scores in this case"

        # prepare the LogitsProcessor
        logit_processor_list = LogitsProcessorList(
            [DepthBasedSequenceBiasLogitsProcessor(self.bias_terms)]
        )
        inputs = self.vanilla_generator.batch_tokenize(inputs, mask=True)
        generate_options_kwargs = dataclasses.asdict(options)
        if isinstance(options, BeamSearchGenerationOptions):
            res = self.vanilla_generator.model.generate(
                input_ids=inputs.ids,
                attention_mask=inputs.mask,
                renormalize_logits=True,  # important,
                logits_processor=logit_processor_list,
                **generate_options_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Generation Options not supported for {options.__class__}"
            )

        # -- For the old version should be compute_transition_beam_scores
        output_mask = torch.where(res.sequences != self.pad_token_id, 1, 0).to(
            self.device
        )
        if self.pad_token_id == self.decoder_start_token_id:
            output_mask[:, 0] = 1
        transition_scores = self.vanilla_generator.model.compute_transition_scores(
            res.sequences,
            res.scores,
            res.beam_indices,
            normalize_logits=False,  # for bs the logits are already normalized
        )
        full_score = torch.sum(transition_scores, dim=-1)
        return FullSequenceGenerationOutput(
            sequences=res.sequences,
            output_mask=output_mask,
            transition_scores=transition_scores,
            all_scores=res.scores,
            sequence_scores=full_score,
        )
