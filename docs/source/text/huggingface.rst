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

Legacy
******

The old huggingface wrappers are listed below for reference, but should not be used
for future development.

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
