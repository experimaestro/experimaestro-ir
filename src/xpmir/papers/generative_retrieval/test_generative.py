from itertools import repeat
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from datamaestro_text.data.ir.base import TextTopic
from experimaestro import Param
from experimaestro.taskglobals import Env as TaskEnv
from experimaestro.xpmutils import DirectoryContext
from xpmir.learning.context import StepTrainingHook

import xpmir.letor.trainers.generative as generative
from xpmir.learning.context import TrainerContext
from xpmir.learning.base import Random
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.learner import Learner
from xpmir.learning.optim import Adam, get_optimizers
from xpmir.letor.samplers import (
    DocumentRecord,
    PairwiseRecord,
    PairwiseSampler,
    TopicRecord,
)
from xpmir.neural.generative import IdentifierGenerator, StepwiseGenerator
from xpmir.utils.iter import RandomSerializableIterator, SerializableIterator


class ProbaTabIdentifierGenerator(IdentifierGenerator):
    """generate the id of the token based on a proba table,
    Assuming that there are only 1 depth"""

    depth: Param[int]
    """Maximum generation length"""

    nb_tokens: Param[int]
    """Number of tokens (excludes eos/pad tokens)"""

    nb_texts: Param[int]
    """The number of the documents and queries to be processed, each one will
    correspond to one vector in each embeddings"""

    def stepwise_iterator(self) -> StepwiseGenerator:
        return ProbaTabStepwiseGenerator(self)

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()

        self._dummy_params = nn.Parameter(torch.Tensor())
        self.text2id = {}

        # Uses the pad_token_id to cater for different lengths
        shape = [self.nb_texts] + [self.nb_tokens + 2 for _ in range(self.depth)]
        self.logits = nn.Parameter(torch.randn(*shape))

        self.eos_token_id = self.nb_tokens
        self.pad_token_id = self.nb_tokens + 1

    @property
    def device(self):
        return self._dummy_params.device

    def log_probabilities(self):
        """Returns the log probabilities for each sequence and document"""
        logits = self.logits
        last_ix = (range(self.nb_texts),) + tuple(
            repeat(self.pad_token_id, logits.ndim - 2)
        )
        return self._log_probabilities(logits, last_ix)

    def _log_probabilities(self, logits: torch.Tensor, last_ix: Tuple[int]):

        if len(last_ix) == 0:
            # End of the recursion: EOS has probability 1
            return {"_": torch.zeros(self.nb_texts)}

        # Get the log-probabilities conditionned
        log_probs = logits[last_ix][:, :-1].log_softmax(dim=1)
        seq_log_probs = {"_": log_probs[:, self.eos_token_id]}
        for ix, log_prob in enumerate(log_probs[:, :-1].transpose(0, 1)):
            subseq_log_probs = self._log_probabilities(logits[:, ix], last_ix[:-1])
            for s, log_prob_s in subseq_log_probs.items():
                seq_log_probs[f"{ix}{s}"] = log_prob + log_prob_s
        return seq_log_probs


def random_derangement(random: np.random.RandomState, n: int):
    remaining = list(range(n))
    selected = [False for _ in range(n)]
    derangement = []

    for j in range(n - 1):
        shift = 0 if selected[j] else 1
        p = random.randint(len(remaining) - shift)

        if ((j == n - 2) and not selected[n - 1]) or (remaining[p] == j):
            p = len(remaining) - 1

        derangement.append(remaining[p])
        selected[remaining[p]] = True
        remaining.pop(p)

    assert len(remaining) == 1
    derangement.append(remaining[0])

    assert all(
        ix != jx for ix, jx in enumerate(derangement)
    ), f"Not a derangement: f{derangement}"
    return derangement


class FakePairwiseSampler(PairwiseSampler):
    nb_doc: Param[int]
    """Number of documents and topics"""

    def initialize(self, random):
        super().initialize(random)
        self.topics = [
            TopicRecord(TextTopic(f"Topic {ix}")) for ix in range(self.nb_doc)
        ]
        self.documents = [
            DocumentRecord(TextTopic(f"Document {ix}")) for ix in range(self.nb_doc)
        ]

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        def iter(random: np.random.RandomState):
            while True:
                derangement = random_derangement(random, self.nb_doc)
                indices = list(range(self.nb_doc))
                random.shuffle(indices)
                for ix in indices:
                    yield PairwiseRecord(
                        self.topics[ix],
                        self.documents[ix],
                        self.documents[derangement[ix]],
                    )

        return RandomSerializableIterator(self.random, iter)


class LoggingHook(StepTrainingHook):
    generator: Param[ProbaTabIdentifierGenerator]
    sampler: Param[FakePairwiseSampler]
    steps: Param[int]

    def get_matrix(
        self, log_p: Dict[str, torch.Tensor], sequences: List[str], texts: List[str]
    ):
        text_ids = [self.generator.text2id[text] for text in texts]

        # Matrix texts x sequence
        texts_logp = torch.Tensor(
            [[log_p[s][text_id] for s in sequences] for text_id in text_ids]
        )

        values = (torch.arange(0, len(sequences)).unsqueeze(0) * texts_logp.exp()).sum(
            1
        )
        _, indices = values.sort()

        texts, log_p = [texts[ix] for ix in indices], texts_logp[indices, :]

        fig, ax = plt.subplots()
        ax.imshow(log_p.exp().numpy(), interpolation="none", aspect="equal")
        ax.set_xticks(np.arange(len(sequences)), labels=sequences)
        ax.set_yticks(np.arange(len(texts)), labels=texts)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        fig.tight_layout()

        return texts, log_p, fig

    def __post_init__(self):
        self.first = True

    def after(self, state: TrainerContext):
        if state.steps % self.steps == 0 or self.first:
            self.first = False
            with torch.no_grad():
                log_p = self.generator.log_probabilities()

            sequences = list(log_p.keys())
            sequences.sort()

            topics, topics_logp, figure = self.get_matrix(
                log_p,
                sequences,
                [record.topic.get_text() for record in self.sampler.topics],
            )
            state.writer.add_figure("topics", figure, state.steps)

            documents, documents_logp, figure = self.get_matrix(
                log_p,
                sequences,
                [record.document.get_text() for record in self.sampler.documents],
            )
            state.writer.add_figure("documents", figure, state.steps)


class ProbaTabStepwiseGenerator(StepwiseGenerator):
    tensors: List[torch.tensor]

    def __init__(self, id_generator: ProbaTabIdentifierGenerator):
        super().__init__()

        # The identifier to use to generate the next step's token
        self.id_generator = id_generator

    def init(self, texts: List[str]) -> torch.Tensor:
        "Transform the texts to id of the embeddings"
        self.tensors = []
        for text in texts:
            text_id = self.id_generator.text2id.setdefault(
                text, len(self.id_generator.text2id)
            )
            assert text_id < self.id_generator.nb_texts
            self.tensors.append(self.id_generator.logits[text_id])

        self.last_ix = tuple(
            repeat(self.id_generator.pad_token_id, len(self.tensors[0].shape) - 1)
        )

    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Return the log_proba"""

        if token_ids is not None:
            assert token_ids.ndim == 1, "Token IDs should be a vector"
            self.tensors = [tensor[ix] for tensor, ix in zip(self.tensors, token_ids)]

        log_probs = [
            tensor[self.last_ix][:-1].log_softmax(0) for tensor in self.tensors
        ]

        log_probs = torch.stack(log_probs)

        self.last_ix = self.last_ix[1:]
        return log_probs


def test_generative(tmp_path: Path):
    """Test a generative loss"""

    NB_TOKENS = 4
    MAX_DEPTH = 2
    NB_DOCS = 16

    STEPS_PER_EPOCH = 16
    MAX_EPOCHS = (8192 * 16) // STEPS_PER_EPOCH

    context = DirectoryContext(tmp_path)

    proba_tab_model = ProbaTabIdentifierGenerator(
        nb_tokens=NB_TOKENS, nb_texts=2 * NB_DOCS, depth=MAX_DEPTH
    )

    sampler = FakePairwiseSampler(nb_doc=NB_DOCS)

    proba_tab_trainer = generative.GenerativeTrainer(
        loss=generative.PairwiseGenerativeRetrievalLoss(
            id_generator=proba_tab_model, max_depth=MAX_DEPTH
        ),
        sampler=sampler,
        batcher=PowerAdaptativeBatcher(),
        batch_size=NB_DOCS,
        hooks=[LoggingHook(generator=proba_tab_model, sampler=sampler, steps=512)],
    )

    # --- Learning

    random = Random(seed=0)

    # The learner trains the model
    learner = Learner(
        # Misc settings
        random=random,
        # How to train the model
        trainer=proba_tab_trainer,
        # The model to train (splade contains all the parameters)
        model=proba_tab_model,
        # Optimization settings
        steps_per_epoch=STEPS_PER_EPOCH,
        optimizers=get_optimizers(Adam()),
        max_epochs=MAX_EPOCHS,
        listeners=[],
    ).instance(context)

    TaskEnv.instance().taskpath = tmp_path
    learner.execute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "test_gen")
    assert folder.is_dir(), f"{folder} is not a directory"
    test_generative(folder)
