import re
import logging
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from experimaestro.compat import cached_property
from experimaestro import Param, Constant, deprecate
from xpmir.distributed import DistributableModel
from xpmir.text.encoders import (
    Encoder,
    TokensEncoder,
    DualTextEncoder,
    TextEncoder,
    TripletTextEncoder,
)
from xpmir.utils.utils import easylog
from xpmir.learning.context import TrainerContext, TrainState
from xpmir.learning.parameters import ParametersIterator

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        DataCollatorForLanguageModeling,
        AutoModelForMaskedLM,
    )
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

from xpmir.letor.records import TokenizedTexts

logger = easylog()
logger.setLevel(logging.INFO)


class BaseTransformer(Encoder):
    """Base transformer class from Huggingface"""

    model_id: Param[str] = "bert-base-uncased"
    """Model ID from huggingface"""

    trainable: Param[bool]
    """Whether BERT parameters should be trained"""

    layer: Param[int] = 0
    """Layer to use (0 is the last, -1 to use them all)"""

    # move this into a hook
    dropout: Param[Optional[float]] = 0
    """Define a dropout for all the layers"""

    CLS: int
    SEP: int

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    @property
    def pad_tokenid(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def automodel():
        return AutoModel

    def __initialize__(self, noinit=False):
        """Initialize the HuggingFace transformer

        Args:
            noinit (bool, optional): True when the weights don't need to be
            loaded. Defaults to False.

            automodel (type, optional): The class
            used to initialize the model. Defaults to AutoModel.
        """
        super().__initialize__()

        config = AutoConfig.from_pretrained(self.model_id)
        if noinit:
            self.model = self.automodel.from_config(config)
        else:
            if self.dropout == 0:
                self.model = self.automodel.from_pretrained(self.model_id)
            else:
                config.hidden_dropout_prob = self.dropout
                config.attention_probs_dropout_prob = self.dropout
                self.model = self.automodel.from_pretrained(
                    self.model_id, config=config
                )

        # Loads the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

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

    def dim(self):
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return self.tokenizer.vocab_size


class TransformerTokensEncoder(BaseTransformer, TokensEncoder):
    """A tokens encoder based on HuggingFace"""

    def forward(self, toks: TokenizedTexts, all_outputs=False):
        outputs = self.model(
            toks.ids.to(self.device),
            attention_mask=toks.mask.to(self.device) if toks.mask is not None else None,
        )
        if all_outputs:
            return outputs
        return outputs.last_hidden_state


@deprecate
class TransformerVocab(TransformerTokensEncoder):
    """Old tokens encoder"""

    pass


class SentenceTransformerTextEncoder(TextEncoder):
    """A Sentence Transformers text encoder"""

    model_id: Param[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def __initialize__(self):
        super().__initialize__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_id)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts)


class OneHotHuggingFaceEncoder(TextEncoder):
    """A tokenizer which encodes the tokens into 0 and 1 vector
    1 represents the text contains the token and 0 otherwise"""

    model_id: Param[str] = "bert-base-uncased"
    """Model ID from huggingface"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    version: Constant[int] = 2

    def __initialize__(self):
        super().__initialize__()
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


@deprecate
class HuggingfaceTokenizer(OneHotHuggingFaceEncoder):
    """The old encoder for one hot"""

    pass


class TransformerEncoder(BaseTransformer, TextEncoder, DistributableModel):
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

    def __initialize__(self):
        self.encoder.__initialize__()

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


class DualTransformerEncoder(BaseTransformer, DualTextEncoder):
    """Encodes the (query, document pair) using the [CLS] token

    maxlen: Maximum length of the query document pair (in tokens) or None if
    using the transformer limit
    """

    maxlen: Param[Optional[int]] = None

    version: Constant[int] = 2

    def forward(self, texts: List[Tuple[str, str]]):
        tokenized = self.batch_tokenize(texts, maxlen=self.maxlen, mask=True)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            kwargs = {}
            if tokenized.token_type_ids is not None:
                kwargs["token_type_ids"] = tokenized.token_type_ids.to(self.device)

            y = self.model(
                tokenized.ids, attention_mask=tokenized.mask.to(self.device), **kwargs
            )

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size


class DualDuoBertTransformerEncoder(BaseTransformer, TripletTextEncoder):
    """Vector encoding of a (query, document, document) triplet

    Be like: [cls] query [sep] doc1 [sep] doc2 [sep]

    """

    maxlen_query: Param[int] = 64
    """Maximum length for the query, the first document and the second one"""

    maxlen_doc: Param[int] = 224
    """Maximum length for the query, the first document and the second one"""

    def __initialize__(self, noinit=False):
        super().__initialize__(noinit)

        # Add an extra token type
        data = self.model.embeddings.token_type_embeddings.weight.data
        if len(data) < 3:
            logger.info("Adding an extra token type in transformer")
            data = torch.cat((data, torch.zeros(1, data.shape[1])))
            self.model.embeddings.token_type_embeddings = nn.Embedding.from_pretrained(
                data, freeze=False
            )

    def batch_tokenize(
        self,
        texts: List[Tuple[str, str, str]],
        batch_first=True,
        mask=False,
    ) -> TokenizedTexts:

        assert batch_first, "Batch first is the only option"

        query = self.tokenizer(
            [triplet[0] for triplet in texts],
            max_length=self.maxlen_query,
            truncation=True,
        )

        document_1 = self.tokenizer(
            [triplet[1] for triplet in texts],
            max_length=self.maxlen_doc,
            truncation=True,
        )

        document_2 = self.tokenizer(
            [triplet[2] for triplet in texts],
            max_length=self.maxlen_doc,
            truncation=True,
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
                + [self.pad_tokenid] * (maxlen - length_factory[index][3])
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


@dataclass
class MLMModelOutput:
    """Format for the output of the model during Masked Language Modeling"""

    logits: torch.LongTensor
    labels: torch.Tensor


class MLMEncoder(BaseTransformer, DistributableModel, DualTextEncoder):
    """Implementation of the encoder for the Masked Language Modeling task"""

    maxlen: Param[Optional[int]] = None

    mlm_probability: Param[float] = 0.2
    """Probability to mask tokens"""

    noinit: Param[bool] = False
    """Whether to start pre-training from scratch or not"""

    datacollator: DataCollatorForLanguageModeling = None

    @property
    def automodel(self):
        return AutoModelForMaskedLM

    def initialize(self):
        super().initialize(self.noinit)
        logger.info("Model initialized")
        self.datacollator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            return_tensors="pt",
        )

    def forward(self, texts: List[str], info: TrainerContext = None) -> MLMModelOutput:
        tokenized = self.batch_tokenize(texts, mask=True)

        masked = self.datacollator.torch_mask_tokens(tokenized.ids.cpu())

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(
                input_ids=masked[0].to(self.device),
                labels=masked[1].to(self.device),
                attention_mask=tokenized.mask.to(self.device),
            )

        # Maybe easier to simply returns the object returned by the BertForMaskedLM?
        return MLMModelOutput(logits=y.logits, labels=masked[1])

    @property
    def dimension(self) -> int:
        return self.config.hidden_size

    def distribute_models(self, update):
        self.model = update(self.model)


class LayerSelector(ParametersIterator):
    """This class can be used to pick some of the transformer layers"""

    # For freezing everything except the embeddings
    re_layer: Param[str] = r"""(?:encoder|transformer)\.layer\.(\d+)\."""

    transformer: Param[BaseTransformer]
    """The model for which layers are selected"""

    pick_layers: Param[int] = 0
    """Counting from the first processing layers (can be negative, i.e. -1 meaning
    until the last layer excluded, etc. / 0 means no layer)"""

    select_embeddings: Param[bool] = False
    """Whether to pick the embeddings layer"""

    select_feed_forward: Param[bool] = False
    """Whether to pick the feed forward of Transformer layers"""

    def __post_init__(self):
        self._re_layer = re.compile(self.re_layer)

    def __validate__(self):
        if (
            not (self.select_embeddings or self.select_feed_forward)
            and self.pick_layers == 0
        ):
            raise AssertionError("The layer selector will select nothing")

    @cached_property
    def nlayers(self):
        count = 0
        for name, _ in self.transformer.model.named_parameters():
            if m := self._re_layer.search(name):
                count = max(count, int(m.group(1)))
        return count

    def should_pick(self, name: str) -> bool:
        if self.select_embeddings and ("embeddings." in name):
            return True

        if self.select_feed_forward and ("intermediate" in name):
            return True

        if self.pick_layers != 0:
            if m := self._re_layer.search(name):
                layer = int(m.group(1))
                if self.pick_layers < 0:
                    return layer <= self.nlayers + self.pick_layers
                return layer < self.pick_layers

        return False

    def iter(self):
        for name, params in self.transformer.model.named_parameters():
            yield f"model.{name}", params, self.should_pick(name)

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, param in self.transformer.model.named_parameters():
                if self.should_freeze(name):
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False


class TransformerTokensEncoderWithMLMOutput(TransformerTokensEncoder):
    """Transformer that output logits over the vocabulary"""

    @property
    def automodel(self):
        return AutoModelForMaskedLM
