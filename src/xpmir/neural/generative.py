from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, T5Config
from experimaestro import Param
from typing import Optional, List
from abc import abstractmethod

import torch
from torch import nn
import numpy as np

from xpmir.learning.optim import Module
from xpmir.letor.records import TokenizedTexts, BaseRecords
from xpmir.distributed import DistributableModel
from xpmir.rankers import AbstractLearnableScorer


# The modified
class CustomOutputT5(T5ForConditionalGeneration):
    """IR scorer based on T5-like models"""

    def __init__(self, config: T5Config, decoder_outdim):
        super().__init__(config)
        self.decoder_outdim = decoder_outdim

        # Modify LM head
        self.lm_head = nn.Parameter(
            nn.Linear(self.lm_head.in_features, self.decoder_outdim, bias=False)
        )

        # Modify the decoder vocabulary
        decoder_embeddings = nn.Embedding(self.decoder_outdim, self.config.d_model)
        self.get_decoder().set_input_embeddings(decoder_embeddings)

    def forward(self):
        pass


class StepwiseGenerator:
    @abstractmethod
    def init(self, batch_size: int) -> torch.Tensor:
        """Returns the distribution over the first generated tokens (BxV)"""
        pass

    @abstractmethod
    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Returns the distribution over next tokens (BxV), given the last
        generates ones (B)"""
        pass


class IdentifierGenerator(Module):
    """generate the id of the token in a stepwise fashion"""

    hf_id: Param[str]
    """The HuggingFace identifier (to configure the model)"""

    def __initialize__(self):
        # Easy and hacky way to get the device
        self._dummy_params = nn.Parameter(torch.Tensor())
        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, use_fast=True)

    @abstractmethod
    def stepwise_iterator(self) -> StepwiseGenerator:
        pass

    @property
    def device(self):
        return self._dummy_params.device


class T5StepwiseGenerator:
    pass


class T5IteratorGenerator(IdentifierGenerator, DistributableModel):
    """generate the id of the token based on t5-based models"""

    decoder_outdim: Param[int] = 12
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding
    """

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()
        self.t5_model = CustomOutputT5(self.config, self.decoder_outdim)

        self.pad_token_id = self.t5_model.generation_config.pad_token_id
        self.decoder_start_token_id = (
            self.t5_model.generation_config.decoder_start_token_id
        )
        self.eos_token_id = self.t5_model.generation_config.eos_token_id

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
        tokenized = self.batch_tokenize(texts, mask=True)
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
                torch.ones((bs, 1), dtype=torch.long) * self.decoder_start_token_id
            )
        else:
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids of the previous layer is not given")

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
        proba = nn.functional.softmax(
            decoder_output.logits[:, -1, :], dim=-1
        )  # shape [bs, decoder_outdim]

        return proba, decoder_output.past_key_values

    def distribute_models(self, update):
        self.t5_model = update(self.t5_model)


class GenerativeRetrievalScorer(AbstractLearnableScorer):
    """A scorer which will be used for the inference of the generative retrieval model,
    and this scorer is not learnable"""

    id_generator: Param[IdentifierGenerator]

    def __initialize__(self):
        # Load the state_dict?
        pass

    def forward(
        self, inputs: "BaseRecords"
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        pass
