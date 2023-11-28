import logging

from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration, T5Config
from experimaestro import Param, LightweightTask
from typing import Optional, List
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
from xpmir.letor.records import TokenizedTexts
from xpmir.distributed import DistributableModel
from xpmir.learning.context import StepTrainingHook, TrainerContext
from xpmir.papers.generative_retrieval.test_generative import FakePairwiseSampler
from . import IdentifierGenerator, StepwiseGenerator, GeneratorForwardOutput


class CustomOutputT5(T5ForConditionalGeneration):
    """T5-based identifier generation

    The class modifies T5 to use a custom vocabulary in the decoder
    """

    def __init__(self, config: T5Config, decoder_outdim):
        # not including the eos and pad
        self.decoder_outdim = decoder_outdim

        # modification of the config according to our needs
        config.pad_token_id = self.decoder_outdim + 1
        config.decoder_start_token_id = self.decoder_outdim + 1
        config.eos_token_id = self.decoder_outdim
        # save
        self.config = config

        super().__init__(self.config)

        # Modify LM head
        self.lm_head = nn.Linear(
            self.lm_head.in_features, self.decoder_outdim + 1, bias=False
        )

        # Make the input embedding has the name of encoder.embed_tokens
        encoder_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.get_encoder().set_input_embeddings(encoder_embeddings)

        self.config.vocab_size = self.decoder_outdim + 1

        # Modify the decoder vocabulary
        decoder_embeddings = nn.Embedding(
            self.decoder_outdim + 2, self.config.d_model, padding_idx=decoder_outdim + 1
        )
        self.get_decoder().set_input_embeddings(decoder_embeddings)

    def forward(self, **kwargs):
        return super().forward(**kwargs)


class T5StepwiseGenerator(StepwiseGenerator):
    def __init__(self, id_generator: IdentifierGenerator):
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


class T5IdentifierGenerator(IdentifierGenerator, DistributableModel):
    """generate the id of the token based on t5-based models"""

    hf_id: Param[str]
    """The HuggingFace identifier (to configure the model)"""

    decoder_outdim: Param[int] = 10
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding, this number
    doesn't include the pad token and the eos token
    """

    def stepwise_iterator(self) -> StepwiseGenerator:
        return T5StepwiseGenerator(self)

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()

        # Easy and hacky way to get the device
        self._dummy_params = nn.Parameter(torch.Tensor())
        self.config = AutoConfig.from_pretrained(self.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, use_fast=True)

        self.model = CustomOutputT5(self.config, self.decoder_outdim)
        self.pad_token_id = self.model.config.pad_token_id
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        self.eos_token_id = self.model.config.eos_token_id

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

        encoder = self.model.get_encoder()
        tokenized = self.batch_tokenize(texts, maxlen=512, mask=True)
        encoder_output = encoder(
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

    def distribute_models(self, update):
        self.model = update(self.model)


class LoadFromT5(LightweightTask):
    """Load parameters from a T5 model"""

    t5_model: Param[T5IdentifierGenerator]
    """the target"""

    def execute(self):
        self.t5_model.initialize(None)

        # Load from checkpoint
        logging.info("Loading hugginface T5 from checkpoint %s", self.t5_model.hf_id)
        # Load the pre-trained model
        t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model.hf_id)

        # Change the state_dict for the lm_head the decoder embedding
        state_dict = t5_model.state_dict()

        del state_dict["lm_head.weight"]

        # use random initialized t5 decoder
        decoder_key_names = [name for name in state_dict.keys() if "decoder" in name]
        for name in decoder_key_names:
            del state_dict[name]

        logging.info("Loading state dict into CustomOutputT5")
        self.t5_model.model.load_state_dict(state_dict, strict=False)


# WIP
class T5LoggingHook(StepTrainingHook):
    """Only support for the two layer version for the moment"""

    generator: Param[IdentifierGenerator]
    sampler: Param[FakePairwiseSampler]
    steps: Param[int]

    def logits_based_on_given_sequence(self, text, choice):
        # sequence just consist of one number, shows which one to select at level 1
        """Calculate the conditioned probability of the sequence based on the
        given sequence"""
        stepwise_generator = self.generator.stepwise_iterator()
        stepwise_generator.init(text)
        first_layer_proba = nn.functional.log_softmax(
            stepwise_generator.step(None), -1
        )  # shape [bs, dec_dim+1]
        decoder_input = (
            torch.tensor([choice], dtype=torch.long)
            .expand(len(text))
            .to(self.generator.device)
        )
        second_layer_proba = nn.functional.log_softmax(
            stepwise_generator.step(decoder_input), -1
        )
        return (
            first_layer_proba[:, choice].unsqueeze(-1) + second_layer_proba
        )  # shape [bs, dec_dim+1]

    def logits_finish_at_beginning(self, text):
        """Calculate the proba of the first generate the _"""
        stepwise_generator = self.generator.stepwise_iterator()
        stepwise_generator.init(text)
        first_layer_proba = nn.functional.log_softmax(stepwise_generator.step(None), -1)
        return first_layer_proba[:, self.generator.eos_token_id].unsqueeze(-1)

    def get_log_proba(self, text):
        res = torch.cat(
            (
                [
                    self.logits_based_on_given_sequence(text, i)
                    for i in range(self.generator.decoder_outdim)
                ]
            ),
            -1,
        )
        return torch.cat((res, self.logits_finish_at_beginning(text)), -1)

    def get_matrix(self, log_proba, sequences, texts):
        fig, ax = plt.subplots()
        ax.imshow(log_proba.exp().cpu().numpy(), cmap="Blues", interpolation="none")
        ax.set_xticks(np.arange(len(sequences)), labels=sequences)
        ax.set_yticks(np.arange(len(texts)), labels=texts)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        return texts, log_proba, fig

    def __post_init__(self):
        self.first = True

    def after(self, state: TrainerContext):
        if state.steps % self.steps == 0 or self.first:
            self.first = False

            query_text = [record.topic.get_text() for record in self.sampler.topics]
            document_text = [
                record.document.get_text() for record in self.sampler.documents
            ]
            with torch.no_grad():
                log_qry = self.get_log_proba(query_text)
                log_doc = self.get_log_proba(document_text)
            sequences = [
                f"{i}{j}"
                for i in range(self.generator.decoder_outdim)
                for j in (list(range(self.generator.decoder_outdim)) + ["_"])
            ] + ["_"]

            _, _, figure = self.get_matrix(
                log_qry,
                sequences,
                query_text,
            )
            state.writer.add_figure("topics", figure, state.steps)

            _, _, figure = self.get_matrix(
                log_doc,
                sequences,
                query_text,
            )
            state.writer.add_figure("documents", figure, state.steps)


# # test
# from xpmir.neural.generative import GenerativeRetrievalScorer

# if __name__ == "__main__":
#     model = T5IdentifierGenerator(hf_id='t5-base')
#     scorer = GenerativeRetrievalScorer(
#         id_generator=model,
#         max_depth=5
#     )

#     scorer = scorer.instance()
#     scorer.initialize(None)

#     res = scorer.rsv(
#         query = "what is your name",
#         scored_documents = ["my name is tom", "hello kitty"]
#     )

#     print(res)
