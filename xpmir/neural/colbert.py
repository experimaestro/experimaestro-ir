import math
from typing import Optional
from experimaestro import param, config, Choices, Param, default
import torch
from torch import nn
from typing_extensions import Annotated
from xpmir.dm.data.base import Index
from . import InteractionScorer
import xpmir.neural.modules as modules


class Colbert(InteractionScorer):
    """ColBERT model

    Implementation of the Colbert model from:

    > Khattab, Omar, and Matei Zaharia.
    > “ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.”
    > SIGIR 2020, Xi'An, China

    For the standard Colbert model, use BERT as the vocab(ulary)

    Attributes:

        compression_size: Projection layer for the last layer (or 0 if None)
    """

    masktoken: Param[bool] = True
    querytoken: Param[bool] = True
    doctoken: Param[bool] = True

    compression_size: Param[int] = 128

    def __validate__(self):
        super().__validate__()
        assert not self.vocab.static(), "The vocabulary should be learnable"

        assert self.compression_size >= 0, "Last layer size should be 0 or above"

        assert not self.masktoken, "Not implemented"
        assert not self.querytoken, "Not implemented"
        assert not self.doctoken, "Not implemented"

    def initialize(self, random):
        super().initialize(random)
        self.simmat = modules.InteractionMatrix(self.vocab.pad_tokenid)

    def _forward(self, inputs):
        simmat, tokq, tokd = self.simmat.encode_query_doc(
            self.vocab, inputs, d_maxlen=self.dlen, q_maxlen=self.qlen
        )

        maxperqtoken, _ = simmat.max(2)
        return maxperqtoken.sum(2).squeeze()
