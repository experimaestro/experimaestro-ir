from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
from experimaestro import Param
from . import IdentifierGenerator, StepwiseGenerator
from xpmir.utils.utils import easylog

logger = easylog()


class ProbaTabStepwiseGenerator(StepwiseGenerator):

    text_ids: torch.tensor
    """The id of the input text, with torch.long of dtype inside, size of [bs]"""

    bs: int
    """The batch size"""

    def __init__(self, id_generator: IdentifierGenerator):
        super().__init__()

        # The identifier to use to generate the next step's token
        self.id_generator = id_generator

    def init(self, texts: List[str]) -> torch.Tensor:
        "Transform the texts to id of the embeddings"
        text_ids = []
        self.bs = len(texts)
        for text in texts:
            if text not in self.id_generator.seen_text.keys():
                self.id_generator.seen_text[text] = len(self.id_generator.seen_text)
            text_ids.append(self.id_generator.seen_text[text])
        self.text_ids = torch.tensor(text_ids).to(self.id_generator.device)

    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Return the log_proba"""
        return self.id_generator(token_ids, self.text_ids)


class ProbaTabIdentifierGeneratorOneLayer(IdentifierGenerator):
    """generate the id of the token based on a proba table,
    Assuming that there are only 1 depth"""

    decoder_outdim: Param[int] = 16
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding, this number
    doesn't include the pad token and the eos token
    """

    nb_docs: Param[int] = 48
    """The number of the documents and queries to be processed, each one will
    correspond to one vector in each embeddings"""

    def stepwise_iterator(self) -> StepwiseGenerator:
        return ProbaTabStepwiseGenerator(self)

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()

        self._dummy_params = nn.Parameter(torch.Tensor())
        self.seen_text = {}

        self.proba_table_l1 = nn.Embedding(self.nb_docs, self.decoder_outdim + 1)
        self.proba_table_l1.weight.data.normal_(0, 1)

        self.pad_token_id = self.decoder_outdim + 1
        self.eos_token_id = self.decoder_outdim

    @property
    def device(self):
        return self._dummy_params.device

    def forward(self, input_ids, text_ids):  # shape [bs] or None  # shape [bs]
        """Return the log_proba of the text give the previous generated tokens"""
        if input_ids is None:
            return nn.functional.log_softmax(self.proba_table_l1(text_ids), dim=-1).to(
                self.device
            )


class ProbaTabIdentifierGeneratorTwoLayers(IdentifierGenerator):
    """generate the id of the token based on a proba table,
    Assuming that there are only 2 depth"""

    decoder_outdim: Param[int] = 4
    """The decoder output dimension for the t5 model, use it to
    rebuild the lm_head and the decoder embedding, this number
    doesn't include the pad token and the eos token
    """

    nb_docs: Param[int] = 48
    """The number of the documents and queries to be processed, each one will
    correspond to one vector in each embeddings"""

    def stepwise_iterator(self) -> StepwiseGenerator:
        return ProbaTabStepwiseGenerator(self)

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()

        self._dummy_params = nn.Parameter(torch.Tensor())
        self.seen_text = {}

        self.proba_table_l1 = nn.Embedding(self.nb_docs, self.decoder_outdim + 1)
        self.proba_table_l1.weight.data.normal_(0, 0.8)
        embeddings = [
            nn.Embedding(self.nb_docs, self.decoder_outdim + 1)
            for _ in range(self.decoder_outdim + 1)
        ]
        for embedding in embeddings:
            embedding.weight.data.normal_(0, 0.8)
        self.proba_table_l2 = nn.ModuleList(embeddings)

        self.pad_token_id = self.decoder_outdim + 1
        self.eos_token_id = self.decoder_outdim

    @property
    def device(self):
        return self._dummy_params.device

    def forward(self, input_ids, text_ids):  # shape [bs] or None
        """Return the log_proba of the text give the previous generated tokens"""
        if input_ids is None:
            return nn.functional.log_softmax(self.proba_table_l1(text_ids), dim=-1).to(
                self.device
            )
        else:
            selected = [self.proba_table_l2[i](j) for i, j in zip(input_ids, text_ids)]
            return nn.functional.log_softmax(torch.stack(selected), dim=-1).to(
                self.device
            )
