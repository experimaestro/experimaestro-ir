"""Returns the TAS-Balanced model (from huggingface)

> Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling
> Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, Allan Hanbury

Returns:
    DotDense: A DotDense ranker based on tas-balanced
"""
from xpmir.text.huggingface import TransformerEncoder, TransformerTextEncoderAdapter
from experimaestro.huggingface import ExperimaestroHFHub
from xpmir.neural.dual import DotDense

encoder = TransformerEncoder(
    model_id="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    trainable=True,
    maxlen=200,
)

model = DotDense(
    encoder=encoder,
    query_encoder=TransformerTextEncoderAdapter(encoder=encoder, maxlen=30),
)

ExperimaestroHFHub(model).push_to_hub(repo_id="xpmir/tas-balanced")
