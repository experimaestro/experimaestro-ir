from typing import List, Optional, Tuple
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.context import Context, InitializationHook
from xpmir.letor import DistributedDeviceInformation
from xpmir.letor.samplers import PairwiseSampler, PairwiseInBatchNegativesSampler
from transformers import AutoModelForMaskedLM
from xpmir.neural import TorchLearnableScorer
from xpmir.text.huggingface import TransformerVocab
from xpmir.text.encoders import TextEncoder
from xpmir.letor.trainers.batchwise import BatchwiseTrainer
from xpmir.neural.dual import DotDense, FlopsRegularizer
from xpmir.utils import easylog

logger = easylog()


class Aggregation(Config):
    """The aggregation function for Splade"""

    pass


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def __call__(self, logits, mask):
        values, _ = torch.max(
            torch.log1p(torch.relu(logits) * mask.to(logits.device).unsqueeze(-1)),
            dim=1,
        )
        return values


class SumAggregation(Aggregation):
    """Aggregate using a sum"""

    def __call__(self, logits, mask):
        return torch.sum(
            torch.log1p(torch.relu(logits) * mask.to(logits.device).unsqueeze(-1)),
            dim=1,
        )


class SpladeTextEncoderModel(nn.Module):
    def __init__(self, encoder, aggregation):
        super().__init__()
        self.encoder = encoder
        self.aggregation = aggregation

    def forward(self, tokenized):
        out = self.encoder(tokenized, all_outputs=True)
        out = self.aggregation(out.logits, tokenized.mask)
        return out


class SpladeTextEncoder(TextEncoder):
    """Splade model

    It is only a text encoder since the we use xpmir.neural.dual.DotDense
    """

    encoder: Param[TransformerVocab]
    """The encoder from Hugging Face"""

    aggregation: Param[Aggregation]
    """How to aggregate the vectors"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    def initialize(self):
        self.encoder.initialize(automodel=AutoModelForMaskedLM)
        self.model = SpladeTextEncoderModel(self.encoder, self.aggregation)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized = self.encoder.batch_tokenize(texts, mask=True, maxlen=self.maxlen)
        out = self.model(tokenized)
        return out

    def static(self):
        return False


class DistributedSpladeTextEncoderHook(InitializationHook):
    """Hook to distribute the model processing

    When in multiprocessing/multidevice, use `torch.nn.parallel.DistributedDataParallel`,
    otherwise use `torch.nn.DataParallel`.
    """

    splade: Param[SpladeTextEncoder]
    """The splade text encoder"""

    def after(self, state: Context):
        info = state.device_information
        if isinstance(info, DistributedDeviceInformation):
            logger.info("Using a distributed model with rank=%d", info.rank)
            self.splade.model = nn.parallel.DistributedDataParallel(
                self.splade.model, device_ids=[info.rank]
            )
        else:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                logger.info(
                    "Setting up DataParallel for Splade text encoder (%d GPUs)", n_gpus
                )
                self.splade.model = torch.nn.DataParallel(self.splade.model)
            else:
                logger.warning("Only one GPU detected, not using data parallel")


def _splade(lambda_q: float, lambda_d: float, aggregation: Aggregation):
    encoder = TransformerVocab(model_id="distilbert-base-uncased", trainable=True)
    doc_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=200
    )
    query_encoder = SpladeTextEncoder(
        aggregation=SumAggregation(), encoder=encoder, maxlen=30
    )
    return DotDense(encoder=doc_encoder, query_encoder=query_encoder), FlopsRegularizer(
        lambda_q=lambda_q, lambda_d=lambda_d
    )


def spladeV1(lambda_q: float, lambda_d: float):
    """Returns the Splade architecture"""
    return _splade(lambda_q, lambda_d, SumAggregation())


def spladeV2(lambda_q: float, lambda_d: float):
    """Returns the Splade v2 architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (arXiv:2109.10086)
    """
    return _splade(lambda_q, lambda_d, MaxAggregation())


def spladev2___(sampler: PairwiseSampler) -> Tuple[BatchwiseTrainer, DotDense]:
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
    return trainer, spladeV2()
