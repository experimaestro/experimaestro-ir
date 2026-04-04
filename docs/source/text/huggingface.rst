Huggingface Transformers
========================

Models
------

Models follow the HuggingFace hierarchy

.. autoxpmconfig:: xpmir.text.huggingface.base.HFMaskedLanguageModel
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModel

Init Tasks
----------

Model loading is handled by init tasks that are passed at submit time.

.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelInitFromID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFFromCheckpoint

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

.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFTokensEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFCLSEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.OneHotHuggingFaceEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.SentenceTransformerTextEncoder

Hooks
-----

.. autoxpmconfig:: xpmir.text.huggingface.encoders.LayerSelector
