from experimaestro.compat import cached_property
from typing import List, Optional, Tuple, Union
import logging
import re
import torch
import torch.nn as nn
from experimaestro import Param, Constant
from xpmir.context import Context, InitializationHook
from xpmir.distributed import DistributableModel
from xpmir.letor import DistributedDeviceInformation
from xpmir.letor.context import InitializationTrainingHook, TrainState
from xpmir.text.encoders import (
    ContextualizedTextEncoder,
    ContextualizedTextEncoderOutput,
    DualTextEncoder,
    TextEncoder,
    TripletTextEncoder,
)
from xpmir.utils.utils import easylog

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

    dropout: Param[Optional[float]] = 0
    """Define a dropout for all the layers"""

    CLS: int  # id=101
    SEP: int  # id=102

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    @property
    def pad_tokenid(self) -> int:
        return self.tokenizer.pad_token_id

    def initialize(self, noinit=False, automodel=AutoModel):
        """Initialize the HuggingFace transformer

        Args:
            noinit (bool, optional): True when the weights don't need to be
            loaded. Defaults to False.

            automodel (type, optional): The class
            used to initialize the model. Defaults to AutoModel.
        """
        super().initialize(noinit=noinit)

        config = AutoConfig.from_pretrained(self.model_id)
        if noinit:
            self.model = automodel.from_config(config)
        else:
            if self.dropout == 0:
                self.model = automodel.from_pretrained(self.model_id)
            else:
                config.hidden_dropout_prob = self.dropout
                config.attention_probs_dropout_prob = self.dropout
                self.model = automodel.from_pretrained(self.model_id, config=config)

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
            r.get("token_type_ids", None),  # if r["token_type_ids"] else None
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


class SentenceTransformerTextEncoder(TextEncoder):
    """A Sentence Transformers text encoder"""

    model_id: Param[str] = "bert-base-uncased"

    def initialize(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts)


class HuggingfaceTokenizer(TextEncoder):
    """A tokenizer which encodes the tokens into 0 and 1 vector
    1 represents the text contains the token and 0 otherwise"""

    model_id: Param[str] = "bert-base-uncased"
    """Model ID from huggingface"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    version: Constant[int] = 2

    def initialize(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.CLS = self._tokenizer.cls_token_id
        self.SEP = self._tokenizer.sep_token_id
        self.PAD = self._tokenizer.pad_token_id
        self._dummy_params = nn.Parameter(torch.Tensor())

    @property
    def device(self):
        return self._dummy_params.device

    @cached_property
    def tokenizer(self):
        return self._tokenizer

    def batch_tokenize(self, texts):
        r = self.tokenizer(
            list(texts),
            max_length=self.maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return r["input_ids"]

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized_ids = self.batch_tokenize(texts)
        batch_size = len(texts)
        x = torch.zeros(batch_size, self.dimension)
        x[torch.arange(batch_size).unsqueeze(-1), tokenized_ids] = 1
        x[:, [self.PAD, self.SEP, self.CLS]] = 0
        return x.to(self.device)

    @property
    def dimension(self):
        return self.tokenizer.vocab_size

    def static(self):
        return False


class IndependentTransformerVocab(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP]"""

    def __call__(self, tokids):
        with torch.set_grad_enabled(self.trainable):
            y = self.model(tokids)

        return y.last_hidden_state


class TransformerEncoder(TransformerVocab, TextEncoder, DistributableModel):
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

    def distribute_models(self, update):
        self.model = update(self.model)


class TransformerTextEncoderAdapter(TextEncoder, DistributableModel):
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

    def distribute_models(self, update):
        self.encoder.model = update(self.encoder.model)


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

    maxlen: Maximum length of the query document pair (in tokens) or None if
    using the transformer limit
    """

    maxlen: Param[Optional[int]] = None

    version: Constant[int] = 2

    def forward(self, texts: List[Tuple[str, str]]):
        tokenized = self.batch_tokenize(texts, maxlen=self.maxlen, mask=True)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(
                tokenized.ids,
                token_type_ids=tokenized.token_type_ids.to(self.device),
                attention_mask=tokenized.mask.to(self.device),
            )

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size

    # def distribute_models(self, update):
    #     self.model = update(self.model)


class DualDuoBertTransformerEncoder(TransformerVocab, TripletTextEncoder):
    """Encoder of the query-document-document pair of the [cls] token
    Be like: [cls]query[sep]doc1[sep]doc2[sep] with 62 tokens for query
    and 223 for each document.
    """

    def initialize(self, noinit=False, automodel=AutoModel):
        super().initialize(noinit, automodel)
        self.model.embeddings.token_type_embeddings = nn.Embedding(3, self.dimension)

    def batch_tokenize(
        self,
        texts: List[Tuple[str, str, str]],
        batch_first=True,
        maxlen=(
            64,
            224,
            224,
        ),  # for query, first document and second document respectively
        mask=False,
    ) -> TokenizedTexts:

        assert batch_first, "Batch first is the only option"

        query = self.tokenizer(
            [triplet[0] for triplet in texts], max_length=maxlen[0], truncation=True
        )

        document_1 = self.tokenizer(
            [triplet[1] for triplet in texts], max_length=maxlen[1], truncation=True
        )

        document_2 = self.tokenizer(
            [triplet[2] for triplet in texts], max_length=maxlen[2], truncation=True
        )

        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        length_factory = (
            []
        )  # [[query_length, document_1_length, document_2_length, total_length],..]
        new_length = []
        maxlen = 0
        batch_size = len(query["input_ids"])

        # calculate the maxlen of the sum for the 3 texts and stock them
        for index in range(batch_size):
            query_length = len(query["input_ids"][index])
            document_1_length = len(document_1["input_ids"][index]) - 1
            document_2_length = len(document_2["input_ids"][index]) - 1
            total_length_at_index = query_length + document_1_length + document_2_length
            if total_length_at_index > maxlen:
                maxlen = total_length_at_index
            length_factory.append(
                [
                    query_length,
                    document_1_length,
                    document_2_length,
                    total_length_at_index,
                ]
            )

        for index in range(batch_size):
            new_input_ids.append(
                query["input_ids"][index]
                + document_1["input_ids"][index][1:]
                + document_2["input_ids"][index][1:]
                + [0] * (maxlen - length_factory[index][3])
            )
            new_attention_mask.append(
                [1] * length_factory[index][3]
                + [0] * (maxlen - length_factory[index][3])
            )
            new_token_type_ids.append(
                [0] * length_factory[index][0]
                + [1] * length_factory[index][1]
                + [2] * length_factory[index][2]
                + [0] * (maxlen - length_factory[index][3])
            )
            new_length.append(length_factory[index][3])

        new_input_ids = torch.Tensor(new_input_ids).type(torch.long)
        new_attention_mask = torch.Tensor(new_attention_mask).type(torch.long)
        new_token_type_ids = torch.Tensor(new_token_type_ids).type(torch.long)
        new_length = torch.Tensor(new_length).type(torch.long)

        return TokenizedTexts(
            None,
            new_input_ids.to(self.device),
            new_length,
            new_attention_mask if mask else None,
            new_token_type_ids.to(self.device),
        )

    def forward(self, texts: List[Tuple[str, str, str]]):

        tokenized = self.batch_tokenize(texts, mask=True)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(
                tokenized.ids,
                token_type_ids=tokenized.token_type_ids,
                attention_mask=tokenized.mask.to(self.device),
            )

        # Assumes that [CLS] is the first token
        # shape of y.last_hidden_state: (1, len(texts), dimension)
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size

    # def distribute_models(self, update):
    #     self.model = update(self.model)


class LayerFreezer(InitializationTrainingHook):
    """This training hook class can be used to freeze some of the transformer layers"""

    RE_LAYER = re.compile(r"""^(?:encoder|transformer)\.layer\.(\d+)\.""")

    transformer: Param[TransformerVocab]
    """The model"""

    freeze_embeddings: Param[bool] = False
    """Whether embeddings should be frozen"""

    frozen: Param[int] = 0
    """Number of frozen layers (can be negative, i.e. -1 meaning until the last
    layer excluded, etc. / 0 means no layer)"""

    def __init__(self):
        self._initialized = False

    def __validate__(self):
        if not self.freeze_embeddings and self.frozen == 0:
            raise AssertionError("The layer freezer would do nothing")

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

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, param in self.transformer.model.named_parameters():
                if self.should_freeze(name):
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False


# TODO: make the class more generic (but involves changing the models or moving
# this to the training part)
class DistributedModelHook(InitializationHook):
    """Hook to distribute the model processing

    When in multiprocessing/multidevice, use
    `torch.nn.parallel.DistributedDataParallel`, otherwise use
    `torch.nn.DataParallel`.
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
