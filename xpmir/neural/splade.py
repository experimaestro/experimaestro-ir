from typing import List, Tuple
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.letor.samplers import PairwiseSampler, PairwiseInBatchNegativesSampler
from xpmir.neural import TorchLearnableScorer
from xpmir.vocab.huggingface import TransformerVocab
from xpmir.vocab.encoders import TextEncoder
from xpmir.letor.trainers.batchwise import BatchwiseTrainer
from xpmir.neural.siamese import DotDense, FlopsRegularizer
from transformers import AutoModelForMaskedLM


class Aggregation(Config):
    """The aggregation function for Splade"""

    pass


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def __call__(self, logits, mask):
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits) * mask.unsqueeze(-1)),
            dim=1,
        )
        return values


class SumAggregation(Aggregation):
    """Aggregate using a sum"""

    def __call__(self, logits, mask):
        return torch.sum(
            torch.log(1 + torch.relu(logits) * mask.unsqueeze(-1)),
            dim=1,
        )


class SpladeTextEncoder(TextEncoder):
    """Splade model (text encoder)"""

    AUTOMODEL_CLASS: AutoModelForMaskedLM
    encoder: Param[TransformerVocab]
    aggregation: Param[Aggregation]

    def initialize(self):
        self.encoder.initialize()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized = self.vocab.batch_tokenize(inputs, maskoutput=True)
        out = self.vocab(input_ids=tokenized.ids, attention_mask=tokenized.mask)
        out = self.aggregation(out)
        return out


def spladeV1():
    """Returns the Splade architecture"""
    return Splade(
        aggregation=LogAggregation(),
        encoder=TransformerVocab(model_id="distilbert-base-cased"),
    )


def spladeV2():
    """Returns the Splade v2 architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (arXiv:2109.10086)
    """
    encoder = SpladeTextEncoder(
        aggregation=SumAggregation(),
        encoder=TransformerVocab(model_id="distilbert-base-cased"),
    )
    return DotDense(encoder=encoder, regularizer=FlopsRegularizer)


def spladev2(sampler: PairwiseSampler) -> Tuple[BatchwiseTrainer, Splade]:
    """Returns the model described in Splade V2"""
    from xpmir.letor.optim import Adam
    from xpmir.letor.trainers.batchwise import SoftmaxCrossEntropy

    ibn_sampler = PairwiseInBatchNegativesSampler(sampler=sampler)
    trainer = BatchwiseTrainer(
        batch_size=124,
        optimizer=Adam(lr=2e-5),
        sampler=ibn_sampler,
        lossfn=SoftmaxCrossEntropy(),
    )

    # Trained with distillation
    return trainer, Splade.v2()
