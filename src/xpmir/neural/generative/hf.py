import logging

from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, T5Config
from experimaestro import Param, LightweightTask
from typing import Optional, List

import torch
from torch import nn
import numpy as np
from xpmir.letor.records import TokenizedTexts
from xpmir.distributed import DistributableModel
from . import IdentifierGenerator, StepwiseGenerator


class CustomOutputT5(T5ForConditionalGeneration):
    """T5-based identifier generation

    The class modifies T5 to use a custom vocabulary in the decoder
    """

    def __init__(self, config: T5Config, decoder_outdim):
        super().__init__(config)
        self.decoder_outdim = decoder_outdim

        # Modify LM head
        self.lm_head = nn.Linear(
            self.lm_head.in_features, self.decoder_outdim, bias=False
        )

        # Modify the decoder vocabulary
        decoder_embeddings = nn.Embedding(self.decoder_outdim, self.config.d_model)
        self.get_decoder().set_input_embeddings(decoder_embeddings)

    def forward(self, **kwargs):
        return super().forward(**kwargs)


class T5StepwiseGenerator(StepwiseGenerator):
    def __init__(self, id_generator: IdentifierGenerator, max_depth: int = 5):
        super().__init__()
        # The identifier to use to generate the next step's token
        self.id_generator = id_generator
        # The maximum step we can go
        self.max_depth = max_depth

    def init(self, texts: List[str]):
        """Initialize some inner states for further iterations, and return
        the initial decoder input tokens"""
        bs = len(texts)

        self.encoder_output, self.attention_mask = self.id_generator.encode(texts)
        self.past_key_values = None
        self.decoder_input_ids = None
        self.unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )
        self.current_depth = 1

    def step(self) -> torch.Tensor:
        """Returns the distribution over next tokens (BxV) by performing a
        stepwise iteration"""
        proba, self.past_key_values = self.id_generator(
            self.attention_mask,
            self.encoder_output,
            self.decoder_input_ids,
            past_key_values=self.past_key_values,
        )
        self.current_depth += 1
        return proba

    def set_token_state(self, new_tokens: torch.LongTensor):
        """Modifying the state of the generator after getting the new generated
        tokens, together with modifying the mask

        input: shape [bs, ]

        """
        # mask some tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        new_tokens = (
            new_tokens * self.unfinished_sequences
            + self.id_generator.pad_token_id * (1 - self.unfinished_sequences)
        )
        # update the tokens
        self.decoder_input_ids = new_tokens.unsqueeze(-1)  # shape [bs, 1]
        # update the mask if encounter eos in the loop
        # it will make the eos position 0 and during the next loop
        # of recursive, it will be skipped
        self.unfinished_sequences = self.unfinished_sequences.mul(
            new_tokens.tile(1, 1)
            .ne(torch.tensor([[self.id_generator.eos_token_id]]))
            .prod(dim=0)
        )

    def get_token_state(self):
        """Return the token state and the unfinished sequences(mask)"""
        return (self.decoder_input_ids, self.unfinished_sequences)

    def stopping_criteria(self) -> bool:
        # end the recursive if all the generation is finish or reaches the max_length
        return (
            self.unfinished_sequences.max() == 0 or self.current_depth == self.max_depth
        )


class T5IdentifierGenerator(IdentifierGenerator, DistributableModel):
    """generate the id of the token based on t5-based models"""

    decoder_outdim: Param[int] = 12
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding
    """

    max_depth: Param[int] = 5
    """The maximum depth of the iterative generation"""

    def stepwise_iterator(self) -> StepwiseGenerator:
        return T5StepwiseGenerator(self, self.max_depth)

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()

        # Easy and hacky way to get the device
        self._dummy_params = nn.Parameter(torch.Tensor())
        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, use_fast=True)

        self.t5_model = CustomOutputT5(self.config, self.decoder_outdim)
        self.pad_token_id = self.t5_model.generation_config.pad_token_id
        self.decoder_start_token_id = (
            self.t5_model.generation_config.decoder_start_token_id
        )
        self.eos_token_id = self.t5_model.generation_config.eos_token_id

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
            r["length"],
            r.get("attention_mask", None),
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
        )

    def encode(self, texts: List[str]):
        """Returns the encoder_output and the input mask for the given text,
        which could accelerate the autoregressive generation procedure"""

        encoder = self.t5_model.get_encoder()
        tokenized = self.batch_tokenize(texts, maxlen=512, mask=True)
        encoder_output = encoder(
            tokenized.ids,
            attention_mask=tokenized.mask.to(self.device),
            return_dict=True,
        )
        return encoder_output, tokenized.mask

    def forward(
        self,
        encoder_attention_mask,  # shape [bs, seq] with 0 or 1
        encoder_outputs,
        decoder_input_ids=None,  # if given, shape [bs, 1]
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

        # Do a forward pass to get the next token
        # returns three not None values:
        # past_key_values, last_hidden_state, encoder_last_hidden_state
        decoder_output = self.t5_model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        proba = nn.functional.softmax(
            decoder_output.logits[:, -1, :], dim=-1
        )  # shape [bs, decoder_outdim]

        return proba, decoder_output.past_key_values

    def distribute_models(self, update):
        self.t5_model = update(self.t5_model)


class LoadFromT5(LightweightTask):
    """Load parameters from a T5 model"""

    model: Param[T5IdentifierGenerator]
    """the target"""

    def execute(self):
        self.model.initialize(None)

        # Load from checkpoint
        logging.info("Loading hugginface T5 from checkpoint %s", self.model.hf_id)
        # Load the pre-trained model
        t5_model = T5ForConditionalGeneration.from_pretrained(self.model.hf_id)

        # Change the state_dict for the lm_head the decoder embedding
        state_dict = t5_model.state_dict()

        del state_dict["lm_head.weight"]

        logging.info("Loading state dict into CustomOutputT5")
        self.model.load_state_dict(state_dict, strict=False)
