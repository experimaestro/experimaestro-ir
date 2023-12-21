Huggingface Transformers
========================

Base
----

.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelConfig
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelConfigId

Models
------

.. autoxpmconfig:: xpmir.text.huggingface.base.HFBaseModel
.. autoxpmconfig:: xpmir.text.huggingface.base.HFMaskedLanguageModel

Tokenizers
----------

.. autoxpmconfig:: xpmir.text.huggingface.base.HFTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFListTokenizer

Encoders
--------

.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFTextEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFTextListTokensEncoder


Legacy
******

.. currentmodule:: xpmir.text.huggingface

.. autoxpmconfig:: BaseTransformer

Encoders
--------

.. autoxpmconfig:: TransformerEncoder
.. autoxpmconfig:: TransformerTokensEncoder
.. autoxpmconfig:: TransformerTextEncoderAdapter

.. autoxpmconfig:: DualTransformerEncoder
.. autoxpmconfig:: SentenceTransformerTextEncoder
.. autoxpmconfig:: OneHotHuggingFaceEncoder
.. autoxpmconfig:: DualDuoBertTransformerEncoder

.. autoxpmconfig:: TransformerVocab

.. autoxpmconfig:: TransformerTokensEncoderWithMLMOutput


Tokenizers
----------

.. autoxpmconfig:: OneHotHuggingFaceEncoder
.. autoxpmconfig:: HuggingfaceTokenizer

Masked-LM
--------=

.. autoxpmconfig:: MLMEncoder

Hooks
-----

.. autoxpmconfig:: LayerSelector
