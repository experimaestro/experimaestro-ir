from experimaestro.compat import cached_property
from typing import ClassVar, List, Optional, Tuple, Union
import logging
import re
import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.context import Context, InitializationHook
from xpmir.letor import DistributedDeviceInformation
from xpmir.letor.context import StepTrainingHook, TrainState
from xpmir.text.encoders import (
    ContextualizedTextEncoder,
    ContextualizedTextEncoderOutput,
    DualTextEncoder,
    TextEncoder,
)
from xpmir.utils import easylog

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

from xpmir.letor.records import TokenizedTexts
import xpmir.text as text

logger = easylog()


class TransformerVocab(text.Vocab):
    """Transformer-based encoder from Huggingface"""

    model_id: Param[str] = "bert-base-uncased"
    """Model ID from huggingface"""

    trainable: Param[bool]
    """Whether BERT parameters should be trained"""

    layer: Param[int] = 0
    """Layer to use (0 is the last, -1 to use them all)"""

    CLS: int
    SEP: int

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    @property
    def pad_tokenid(self) -> int:
        return self.tokenizer.pad_token_id

    def initialize(self, noinit=False, automodel=AutoModel):
        super().initialize(noinit=noinit)

        if noinit:
            config = AutoConfig.from_pretrained(self.model_id)
            self.model = automodel.from_config(config)
        else:
            self.model = automodel.from_pretrained(self.model_id)

        # Loads the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        layer = self.layer
        if layer == -1:
            layer = None
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id

        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

    def parameters(self, recurse=True):
        if self.trainable:
            return super().parameters(recurse)
        return []

    def train(self, mode: bool = True):
        # We should not make this layer trainable unless asked
        if mode:
            if self.trainable:
                self.model.train(mode)
        else:
            self.model.train(mode)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def tok2id(self, tok):
        return self.tokenizer.vocab[tok]

    def static(self):
        return not self.trainable

    def batch_tokenize(
        self,
        texts: Union[List[str], List[Tuple[str, str]]],
        batch_first=True,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
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
        )

    def id2tok(self, idx):
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        # return self.tokenizer.ids_to_tokens[idx]
        return self.tokenizer.id_to_token(idx)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()

    def maxtokens(self) -> int:
        return self.tokenizer.model_max_length

    def forward(self, toks: TokenizedTexts, all_outputs=False):
        outputs = self.model(
            toks.ids.to(self.device),
            attention_mask=toks.mask.to(self.device) if toks.mask is not None else None,
        )
        if all_outputs:
            return outputs
        return outputs.last_hidden_state

    def dim(self):
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return self.tokenizer.vocab_size


class IndependentTransformerVocab(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP]"""

    def __call__(self, tokids):
        with torch.set_grad_enabled(self.trainable):
            y = self.model(tokids)

        return y.last_hidden_state


class TransformerEncoder(TransformerVocab, TextEncoder):
    """Encodes using the [CLS] token"""

    maxlen: Param[Optional[int]] = None

    def forward(self, texts: List[str], maxlen=None):
        tokenized = self.batch_tokenize(texts, maxlen=maxlen or self.maxlen, mask=True)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(tokenized.ids, attention_mask=tokenized.mask.to(self.device))

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self):
        return self.dim()

    def with_maxlength(self, maxlen: int):
        return TransformerTextEncoderAdapter(encoder=self, maxlen=maxlen)


class TransformerTextEncoderAdapter(TextEncoder):
    encoder: Param[TransformerEncoder]
    maxlen: Param[Optional[int]] = None

    def initialize(self):
        self.encoder.initialize()

    @property
    def dimension(self):
        return self.encoder.dimension

    def forward(self, texts: List[str], maxlen=None):
        return self.encoder.forward(texts, maxlen=self.maxlen)

    def static(self):
        return self.encoder.static()

    @property
    def vocab_size(self):
        return self.encoder.vocab_size


class ContextualizedTransformerEncoder(TransformerVocab, ContextualizedTextEncoder):
    """Returns the contextualized output at the various layers"""

    @property
    def dimension(self):
        return self.dim()

    def forward(
        self,
        texts: List[str],
        maxlen=None,
        only_tokens=True,
        output_hidden_states=False,
    ):
        tokenized = self.batch_tokenize(texts, maxlen=maxlen or self.maxlen, mask=True)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
                output_hidden_states=output_hidden_states,
            )
            mask = tokenized.mask
            ids = tokenized.ids.to(self.device)
            if only_tokens:
                mask = (self.CLS != ids) & (self.SEP != ids) & mask.to(self.device)
            return ContextualizedTextEncoderOutput(
                ids, mask, y.last_hidden_state, y.hidden_states
            )

    def with_maxlength(self, maxlen: int):
        return ContextualizedTextEncoderAdapter(encoder=self, maxlen=maxlen)


class ContextualizedTextEncoderAdapter(ContextualizedTextEncoder):
    encoder: Param[ContextualizedTransformerEncoder]
    maxlen: Param[Optional[int]] = None

    def initialize(self):
        self.encoder.initialize()

    def forward(self, texts: List[str], **kwargs):
        return self.encoder.forward(texts, maxlen=self.maxlen, **kwargs)

    @property
    def dimension(self):
        return self.encoder.dimension

    def static(self):
        return self.encoder.static()

    @property
    def vocab_size(self):
        return self.encoder.vocab_size


class DualTransformerEncoder(TransformerVocab, DualTextEncoder):
    """Encodes the (query, document pair) using the [CLS] token

    maxlen: Maximum length of the query document pair (in tokens) or None if using the transformer limit
    """

    maxlen: Param[Optional[int]] = None

    def forward(self, texts: List[Tuple[str, str]]):
        tokenized = self.batch_tokenize(texts, maxlen=self.maxlen, mask=True)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(tokenized.ids, attention_mask=tokenized.mask.to(self.device))

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size


class LayerFreezer(StepTrainingHook):
    """This training hook class can be used to freeze some of the transformer layers"""

    RE_LAYER = re.compile(r"""^(?:encoder|transformer)\.layer\.(\d+)\.""")

    transformer: Param[TransformerVocab]
    """The model"""

    freeze_embeddings: Param[bool] = True

    frozen: Param[int]
    """Number of frozen layers (can be negative, i.e. -1 meaning until the last layer excluded, etc. / 0 means no layer)"""

    def __init__(self):
        self._initialized = False

    @cached_property
    def nlayers(self):
        count = 0
        for name, param in self.transformer.model.named_parameters():
            if m := LayerFreezer.RE_LAYER.match(name):
                count = max(count, int(m.group(1)))
        return count

    def should_freeze(self, name: str):
        if self.freeze_embeddings and name.startswith("embeddings."):
            return True

        if self.frozen != 0:
            if m := LayerFreezer.RE_LAYER.match(name):
                layer = int(m.group(1))
                if self.frozen < 0:
                    return layer <= self.nlayers + self.frozen
                return layer < self.frozen

        return False

    def before(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, param in self.transformer.model.named_parameters():
                if self.should_freeze(name):
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False


class DistributedModelHook(InitializationHook):
    """Hook to distribute the model processing

    When in multiprocessing/multidevice, use `torch.nn.parallel.DistributedDataParallel`,
    otherwise use `torch.nn.DataParallel`.
    """

    transformer: Param[TransformerVocab]
    """The model"""

    def after(self, state: Context):
        info = state.device_information
        if isinstance(info, DistributedDeviceInformation):
            logger.info("Using a distributed model with rank=%d", info.rank)
            self.transformer.model = nn.parallel.DistributedDataParallel(
                self.transformer.model, device_ids=[info.rank]
            )
        else:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                logger.info("Setting up DataParallel for transformer (%d GPUs)", n_gpus)
                self.transformer.model = torch.nn.DataParallel(self.transformer.model)
