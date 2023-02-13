from typing import Optional
from xpmir.text.huggingface import TransformerTokensEncoder
from xpmir.neural.splade import SpladeTextEncoder, MaxAggregation, DotDense
from experimaestro.huggingface import ExperimaestroHFHub


def push_splade(
    variant: str, model_id: str, *, query_model_id: Optional[str] = None, config=None
):
    encoder = TransformerTokensEncoder(model_id=model_id, trainable=True)
    query_encoder = (
        TransformerTokensEncoder(model_id=query_model_id, trainable=True)
        if query_model_id
        else None
    )
    aggregation = MaxAggregation()

    # make use the output of the BERT and do an aggregation
    doc_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=200
    )
    query_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=31
    )

    model = DotDense(encoder=doc_encoder, query_encoder=query_encoder)

    ExperimaestroHFHub(model, variant=variant).push_to_hub(
        repo_id="xpmir/splade", config=config
    )


config = {
    "variants": [
        "cocondenser-selfdistil",
        "cocondenser-ensembledistil",
        "efficient-V-large-doc",
        "efficient-VI-BT-large-doc",
    ]
}

push_splade(
    "cocondenser-selfdistil", "naver/splade-cocondenser-selfdistil", config=config
)
push_splade("cocondenser-ensembledistil", "naver/splade-cocondenser-ensembledistil")
push_splade(
    "efficient-V-large-doc",
    "naver/efficient-splade-V-large-doc",
    query_model_id="naver/efficient-splade-V-large-query",
)
push_splade(
    "efficient-VI-BT-large-doc",
    "naver/efficient-splade-VI-BT-large-doc",
    query_model_id="naver/efficient-splade-VI-BT-large-query",
)
