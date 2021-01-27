from typing import List
import torch
import torch.nn as nn

from experimaestro import config, Param
from xpmir.letor.samplers import Records, SamplerRecord
from xpmir.rankers import LearnableScorer, ScoredDocument
from xpmir.vocab import Vocab


@config()
class EmbeddingScorer(LearnableScorer, nn.Module):
    """A scorer based on token embeddings

    Attributes:
        vocab: The embedding model -- the vocab also defines how to tokenize text
        qlen: Maximum query length
        dlen: Maximum document length
        add_runscore:
            Whether the base predictor score should be added to the
            model score
    """

    vocab: Param[Vocab]
    qlen: Param[int] = 20
    dlen: Param[int] = 2000
    add_runscore: Param[bool] = False

    def initialize(self, random):
        self.random = random
        seed = self.random.randint((2 ** 32) - 1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if self.add_runscore:
            self.runscore_alpha = torch.nn.Parameter(torch.full((1,), -1.0))
        self.vocab.initialize()

    def _pad(self, sequences):
        return (
            self.vocab.pad_sequence(sequences, batch_first=True),
            torch.LongTensor([len(s) for s in sequences]),
        )

    def rsv(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        # Prepare the inputs and call the model
        inputs = Records()
        for doc in documents:
            assert doc.content is not None
            inputs.add(SamplerRecord(query, doc.docid, doc.content, doc.score, None))

        with torch.no_grad():
            scores = self(inputs).cpu().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i in range(len(documents)):
            scoredDocuments.append(ScoredDocument(documents[i].docid, scores[i]))

        return scoredDocuments

    def forward(self, inputs: Records):
        # Prepare inputs
        inputs.queries_toks = [self.vocab.tokenize(query) for query in inputs.queries]
        inputs.queries_tokids, inputs.query_len = self._pad(
            [[self.vocab.tok2id(t) for t in tok] for tok in inputs.queries_toks]
        )

        inputs.docs_tokids, inputs.docs_len = self._pad(
            [
                [
                    self.vocab.tok2id(t)
                    for _, t in zip(range(self.dlen), self.vocab.tokenize(document))
                ]
                for document in inputs.documents
            ]
        )

        # Forward to model
        result = self._forward(inputs)

        if len(result.shape) == 2 and result.shape[1] == 1:
            result = result.reshape(result.shape[0])

        # Add run score if needed
        if self.add_runscore:
            alpha = torch.sigmoid(self.runscore_alpha)
            result = alpha * result + (1 - alpha) * inputs.scores

        return result

    def _forward(self, **inputs):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
