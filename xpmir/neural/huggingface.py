from xpmir.letor.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from experimaestro import Param


class HFCrossScorer(LearnableScorer):

    hf_id: Param[str]
    """the id for the huggingface model"""

    max_length: Param[int] = None
    """the max length for the transformer model"""

    def __post_init__(self):
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            self.logger.error(
                "Sentence transformer is not installed:"
                "pip install -U sentence_transformers"
            )
            raise
        self.model = CrossEncoder(self.hf_id, max_length=self.max_length)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        predict_score_list = self.model.predict(
            [(q.text, d.text) for q, d in zip(inputs.queries, inputs.documents)],
            convert_to_tensor=True,
        )  # Tensor[float] of length records size
        return predict_score_list
