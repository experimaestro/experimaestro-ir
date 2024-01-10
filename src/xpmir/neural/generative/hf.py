import logging
import dataclasses
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, T5Config
from experimaestro import Param, LightweightTask
from typing import Optional, List, NamedTuple, Tuple

import torch
from torch import nn
from xpmir.learning import ModuleInitOptions, ModuleInitMode
from xpmir.text.encoders import TokenizedTexts
from xpmir.distributed import DistributableModel
from . import (
    ConditionalGenerator,
    GenerateOptions,
    BeamSearchGenerationOptions,
    StepwiseGenerator,
)


class GeneratorForwardOutput(NamedTuple):
    """The forward output of the generative retrieval"""

    logits: torch.tensor
    past_key_values: Optional[torch.tensor] = None


class FullSequenceGenerationOutput(NamedTuple):
    """The output for the generate method"""

    sequences: torch.tensor
    """The returned sequence
    shape: [bs*num_sequence, max_depth]"""

    output_mask: torch.tensor
    """A mask for the output sequences
    shape: [bs*num_sequence, max_depth]"""

    transition_scores: Optional[torch.tensor] = None
    """The condtional proba for tokens in the sequences, log, normalized
    shape: [bs * num_sequence, max_depth]"""

    all_scores: Optional[Tuple[torch.tensor]] = None
    """All the probabilities, log, normalized, tuple of length max_depth
    each tensor of the tuple has the shape of [bs * num_sequence, vs]"""

    sequence_scores: Optional[torch.tensor] = None
    """The proba for the full sequence, log
    shape: [bs * num_sequence]"""


class T5ConditionalGenerator(ConditionalGenerator, DistributableModel):

    hf_id: Param[str]
    """The HuggingFace identifier (to configure the model)"""

    def stepwise_iterator(self) -> StepwiseGenerator:
        return T5StepwiseGenerator(self)

    def __initialize__(self, options: ModuleInitOptions):
        assert options.mode != ModuleInitMode.RANDOM, "Random mode not handled (yet)"

        super().__initialize__(options)

        # Easy and hacky way to get the device
        self._dummy_params = nn.Parameter(torch.Tensor())

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, use_fast=True)
        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.model = self.initialize_model(options)

        self.pad_token_id = self.model.config.pad_token_id
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.eos_token_id = self.model.config.eos_token_id

        self.encoder = self.model.get_encoder()

    def initialize_model(self, options: ModuleInitOptions):
        return T5ForConditionalGeneration(self.config)

    @property
    def device(self):
        return self._dummy_params.device

    def batch_tokenize(
        self,
        texts: List[str],
        batch_first=True,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        """Tokenize the input text"""
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        else:
            maxlen = min(maxlen, self.tokenizer.model_max_length)

        assert batch_first, "Batch first is the only option"

        r = self.tokenizer(
            list(texts),
            max_length=maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=mask,
        )
        return TokenizedTexts(
            None,
            r["input_ids"].to(self.device),
            r.get("length", None),
            r["attention_mask"].to(self.device) if mask else None,
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
        )

    def encode(self, texts: List[str]):
        """Returns the encoder_output and the input mask for the given text,
        which could accelerate the autoregressive generation procedure"""

        tokenized = self.batch_tokenize(texts, maxlen=512, mask=True)
        encoder_output = self.encoder(
            tokenized.ids,
            attention_mask=tokenized.mask,
            return_dict=True,
        )
        return encoder_output, tokenized.mask

    def forward(
        self,
        encoder_attention_mask,  # shape [bs, seq] with 0 or 1
        encoder_outputs,
        decoder_input_ids=None,  # if given, shape [bs]
        past_key_values=None,
    ):
        """Get the logits from the decoder"""
        bs = encoder_outputs.last_hidden_state.shape[0]

        if past_key_values is None:
            decoder_input_ids = (
                torch.ones((bs, 1), dtype=torch.long).to(self.device)
                * self.decoder_start_token_id
            )
        else:
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids of the previous layer is not given")
            else:
                decoder_input_ids = decoder_input_ids.unsqueeze(-1)

        # Do a forward pass to get the next token
        # returns three not None values:
        # past_key_values, last_hidden_state, encoder_last_hidden_state
        decoder_output = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = decoder_output.logits[:, -1, :]  # shape [bs, decoder_outdim+1]

        return GeneratorForwardOutput(
            logits=logits, past_key_values=decoder_output.past_key_values
        )

    def generate(
        self, inputs: List[str], options: GenerateOptions = None
    ) -> FullSequenceGenerationOutput:
        inputs = self.batch_tokenize(inputs, mask=True)
        generate_options_kwargs = dataclasses.asdict(options)
        if isinstance(options, BeamSearchGenerationOptions):
            res = self.model.generate(
                input_ids=inputs.ids,
                attention_mask=inputs.mask,
                **generate_options_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Generation Options not supported for {options.__class__}"
            )

        if options.return_dict_in_generate:
            output_mask = torch.where(res.sequences != self.pad_token_id, 1, 0).to(
                self.device
            )

            if self.pad_token_id == self.decoder_start_token_id:
                output_mask[:, 0] = 1

            if not options.output_scores:
                return FullSequenceGenerationOutput(
                    sequences=res.sequences,
                    output_mask=output_mask,
                )
            else:
                # -- For the old version should be compute_transition_beam_scores
                transition_scores = self.model.compute_transition_scores(
                    res.sequences,
                    res.scores,
                    res.beam_indices,
                    normalize_logits=False,  # for bs the logits are already normalized
                )
                sequence_scores = torch.sum(transition_scores, dim=-1)
                return FullSequenceGenerationOutput(
                    sequences=res.sequences,
                    output_mask=output_mask,
                    transition_scores=transition_scores,
                    sequence_scores=sequence_scores,
                )
        else:
            output_mask = torch.where(res != self.pad_token_id, 1, 0).to(self.device)

            if self.pad_token_id == self.decoder_start_token_id:
                output_mask[:, 0] = 1

            return FullSequenceGenerationOutput(
                sequences=res,
                output_mask=output_mask,
            )

    def batch_decode(self, generate_output: FullSequenceGenerationOutput) -> List[str]:
        """Decode the sequences to meaningful texts"""
        return self.tokenizer.batch_decode(
            generate_output.sequences, skip_special_tokens=True
        )

    def distribute_models(self, update):
        self.encoder = update(self.model.get_encoder())
        self.model = update(self.model)


class T5StepwiseGenerator(StepwiseGenerator):
    def __init__(self, id_generator: ConditionalGenerator):
        super().__init__()
        # The identifier to use to generate the next step's token
        self.id_generator = id_generator

    def init(self, texts: List[str]):
        """Initialize some inner states for further iterations, and return
        the initial decoder input tokens"""
        self.encoder_output, self.attention_mask = self.id_generator.encode(texts)
        self.past_key_values = None

    def step(self, decoder_input_tokens) -> torch.Tensor:  # input shape [bs]
        """Returns the distribution over next tokens (BxV) by performing a
        stepwise iteration"""
        forward_output: GeneratorForwardOutput = self.id_generator(
            self.attention_mask,
            self.encoder_output,
            decoder_input_tokens,
            past_key_values=self.past_key_values,
        )
        self.past_key_values = forward_output.past_key_values
        return forward_output.logits


class T5ForIdentifierGeneration(T5ForConditionalGeneration):
    """T5-based identifier generation

    The class modifies T5 to use a custom vocabulary in the decoder
    """

    def __init__(self, config: T5Config, decoder_outdim: int):
        # not including the eos and pad
        self.decoder_outdim = decoder_outdim

        # modification of the config according to our needs
        config.pad_token_id = self.decoder_outdim + 1
        config.decoder_start_token_id = self.decoder_outdim + 1
        config.eos_token_id = self.decoder_outdim

        # Keep config at hand
        self.config = config

        super().__init__(self.config)

        # Modify LM head
        self.lm_head = nn.Linear(
            self.lm_head.in_features, self.decoder_outdim + 1, bias=False
        )

        # We have one more token (PAD when )
        encoder_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.config.vocab_size = self.decoder_outdim + 1
        self.get_encoder().set_input_embeddings(encoder_embeddings)

        # Modify the decoder vocabulary
        decoder_embeddings = nn.Embedding(
            self.decoder_outdim + 2, self.config.d_model, padding_idx=decoder_outdim + 1
        )
        self.get_decoder().set_input_embeddings(decoder_embeddings)

    def forward(self, **kwargs):
        return super().forward(**kwargs)


class T5IdentifierGenerator(T5ConditionalGenerator):
    """generate the id of the token based on t5-based models"""

    decoder_outdim: Param[int] = 10
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding, this number
    doesn't include the pad token and the eos token
    """

    def initialize_model(self, options: ModuleInitOptions):
        return T5ForIdentifierGeneration(self.config, self.decoder_outdim)


class T5ForConditionalCustomGeneration(T5ForConditionalGeneration):
    """T5-based model with custom output"""

    def __init__(self, config: T5Config, decoder_outdim: int):
        # not including the eos and pad
        self.config = config
        self.decoder_outdim = decoder_outdim
        config.decoder_start_token_id = self.decoder_outdim - 1

        super().__init__(self.config)

        # Modify LM head
        self.lm_head = nn.Linear(
            self.lm_head.in_features, self.decoder_outdim, bias=False
        )

        encoder_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.get_encoder().set_input_embeddings(encoder_embeddings)

        # Modify the decoder vocabulary
        self.config.vocab_size = self.decoder_outdim
        decoder_embeddings = nn.Embedding(self.decoder_outdim, self.config.d_model)
        self.get_decoder().set_input_embeddings(decoder_embeddings)

    def forward(self, **kwargs):
        return super().forward(**kwargs)


class T5CustomOutputGenerator(T5ConditionalGenerator):
    """generate the id of the token based on t5-based models"""

    #: List of tokens for the output
    tokens: Param[List[str]]

    def initialize_model(self, options: ModuleInitOptions):
        return T5ForConditionalCustomGeneration(self.config, len(self.tokens))


class LoadFromT5(LightweightTask):
    """Load parameters from a T5 model"""

    t5_model: Param[T5ConditionalGenerator]
    """the target"""

    def execute(self):
        self.t5_model.initialize(ModuleInitMode.DEFAULT.to_options())

        # Load from checkpoint
        logging.info("Loading hugginface T5 from checkpoint %s", self.t5_model.hf_id)

        # Load the T5 pre-trained model
        t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model.hf_id)

        # Change the state_dict for the lm_head the decoder embedding
        state_dict = t5_model.state_dict()

        if isinstance(self.t5_model, T5IdentifierGenerator):
            # Just forget about the weights
            del state_dict["lm_head.weight"]

            # use random initialized t5 decoder
            decoder_key_names = [
                name for name in state_dict.keys() if "decoder" in name
            ]
            for name in decoder_key_names:
                del state_dict[name]
        elif isinstance(self.t5_model, T5CustomOutputGenerator):
            # Get the token embeddings from the tokenizer
            token_ids = []
            for token in self.t5_model.tokens:
                ids = self.t5_model.tokenizer.encode(token, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Token {token} is made of {len(ids)} subtokens")
                token_ids.append(ids[0])

            # And restrict our dictionary to the possible tokens
            state_dict["lm_head.weight"] = t5_model.lm_head.weight.detach()[
                (tuple(token_ids),)
            ]
            state_dict[
                "decoder.embed_tokens.weight"
            ] = t5_model.lm_head.weight.detach()[(tuple(token_ids),)]

        logging.info("Loading state dict into the custom T5")
        self.t5_model.model.load_state_dict(state_dict, strict=False)
