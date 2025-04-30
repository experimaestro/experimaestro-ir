from typing import List

from datamaestro.record import Record
from experimaestro import Param
from datamaestro_text.data.ir import TextItem
from xpmir.utils.convert import Converter

from .encoders import InputType, RepresentationOutput, TokenizedTextEncoderBase


class MeanTextEncoder(TokenizedTextEncoderBase[InputType, RepresentationOutput]):
    """Returns the mean of the word embeddings"""

    encoder: Param[TokenizedTextEncoderBase[InputType, RepresentationOutput]]

    def __initialize__(self, options):
        self.encoder.__initialize__(options)

    def static(self):
        return self.encoder.static()

    @property
    def dimension(self):
        return self.encoder.dimension

    def forward(self, texts: List[InputType], options=None) -> RepresentationOutput:
        # emb_texts = self.encoder(texts, options=options)
        emb_texts = self.encoder(texts, options=options)
        # Computes the mean over the time dimension (vocab output is batch x time x dim)
        emb_texts.value = emb_texts.value.mean(1)
        return emb_texts


class TopicTextConverter(Converter[Record, str]):
    """Extracts the text from a topic"""

    def __call__(self, input: Record) -> str:
        return input[TextItem].text
