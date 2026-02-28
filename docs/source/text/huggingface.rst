Huggingface Transformers
========================

Base
----

Models architectures and default parameters are specificied using
a `HFModelConfig`.

.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelConfig
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelConfigFromId

Models
------

Models follow the HuggingFace hierarchy

.. autoxpmconfig:: xpmir.text.huggingface.base.HFMaskedLanguageModel
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModel

Tokenizers
----------

.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizerBase

.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFListTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFStringTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizerAdapter

Encoders
--------


.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFEncoderBase
    :members: from_pretrained_id

.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFTokensEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFCLSEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.OneHotHuggingFaceEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.SentenceTransformerTextEncoder

Hooks
-----

.. autoxpmconfig:: xpmir.text.huggingface.encoders.LayerSelector
