Pre-trained models
==================

XPMIR pre-trained models are published on the
`HuggingFace Hub <https://huggingface.co/models?library=xpmir>`_ under the
``xpmir`` library tag. They can be loaded with
:class:`~xpmir.models.AutoModel` for use in experiments or direct inference.

.. contents:: On this page
   :local:
   :depth: 2


Using existing models
---------------------

Download a model from the Hub using
:meth:`~xpmir.models.AutoModel.load_from_hf_hub`. Thanks to the experimaestro
framework, you can either re-use models inside experiments (with full parameter
tracking) or in pure inference mode.

As experimental models
^^^^^^^^^^^^^^^^^^^^^^

In this mode, the loaded model is an experimaestro configuration that can be
composed with other components (e.g. for comparison, distillation, or pipeline
integration)::

    from xpmir.models import AutoModel

    # Model that can be re-used in experiments
    model = AutoModel.load_from_hf_hub("xpmir/monobert")

Pure inference mode
^^^^^^^^^^^^^^^^^^^

In this mode, the model is instantiated immediately and can score
documents right away::

    from xpmir.models import AutoModel

    # Use this if you want to actually use the model
    model = AutoModel.load_from_hf_hub("xpmir/monobert", as_instance=True)
    model.initialize(None)
    model.rsv("walgreens store sales average", "The average Walgreens salary ranges...")


Cross-encoders
--------------

Cross-encoder models can also be created from any HuggingFace transformer
checkpoint trained for sequence classification, using
:meth:`~xpmir.models.AutoModel.cross_encoder_model`.


Dense models
------------

:class:`~xpmir.neural.dual.Dense` models can be created from any
`Sentence Transformers <https://huggingface.co/models?library=sentence-transformers>`_
checkpoint using :meth:`~xpmir.models.AutoModel.sentence_scorer`.
