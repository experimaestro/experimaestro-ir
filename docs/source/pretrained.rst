Pre-trained models
==================

Experimaestro specific pre-trained models can be found on
the `HuggingFace Hub <https://huggingface.co/models?library=xpmir>`_
searching the `xpmir` library. Models can then be loaded using


Using existing models
---------------------

You can simply download a model from the Hub using `xpmir.models.AutoModel`.
Thanks to the `experimaestro framework <https://github.com/experimaestro/experimaestro-python>`_,
you can either use models in your own experiments or in pure inference mode using
:py:meth:`load_from_hf_hub() <xpmir.models.AutoModel.load_from_hf_hub>`

As experimental models
----------------------

In this mode, you can reuse the model in your experiments -- e.g. to compare this model
with your own, or using it in a complex IR pipeline (e.g. distillation). Please
refer to the `experimaestro-IR documentation <https://experimaestro-ir.readthedocs.io/>_`
for more details::

    from xpmir.models import AutoModel

    # Model that can be re-used in experiments
    model = AutoModel.load_from_hf_hub("xpmir/monobert")

Pure inference mode
-------------------

In this mode, the model can be used right away to score documents::

    from xpmir.models import AutoModel

    # Use this if you want to actually use the model
    model = AutoModel.load_from_hf_hub("xpmir/monobert", as_instance=True)
    model.initialize(None)
    model.rsv("walgreens store sales average", "The average Walgreens salary ranges...")


Cross-encoders
--------------

Cross-encoders models can also be created from any transformer model that has been trained
to classify a query/document using :py:meth:`cross_encoder_model <xpmir.models.AutoModel.cross_encoder_model>`




Dense models
------------

:py:class:`Dense <xpmir.neural.dual.Dense>` models can also be created from
transformers from the Sentence Transformers library (`HuggingFace Hub list <https://huggingface.co/models?library=sentence-transformers>`_) using :py:meth:`sentence_scorer <xpmir.models.AutoModel.sentence_scorer>`.
