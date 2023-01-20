Pre-trained models
==================

Experimaestro specific pre-trained models can be found on
the `HuggingFace Hub <https://huggingface.co/models?library=xpmir>`_
searching the `xpmir` library. Models can then be loaded using::

    from xpmir.models import AutoModel
    from xpmir.letor.records import PointwiseRecords
    hf_id = "xpmir/splade"
    variant = "cocondenser-selfdistil"

    # The model is an experimaestro configuration
    # it can be used in experiments...
    model = AutoModel.load_from_hf_hub(hf_id, variant=variant)

    # ... or directly
    model = model.instance()
    model.initialize(None)
    scores = model(PointwiseRecords.from_texts(["my query"], ["my document"]))

Dense models
------------

:py:class:`Dense <xpmir.neural.dual.Dense>` models can also be created from
transformers from the Sentence Transformers library (`HuggingFace Hub list <https://huggingface.co/models?library=sentence-transformers>`_) using :py:meth:`from_sentence_transformers <xpmir.neural.dual.Dense.from_sentence_transformers>`
