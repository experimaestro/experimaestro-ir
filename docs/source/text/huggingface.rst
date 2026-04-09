HuggingFace Transformers
========================

Integration with `HuggingFace Transformers <https://huggingface.co/docs/transformers/>`_
for loading pre-trained language models, tokenizers, and building transformer-based
text encoders. These components are used by neural models such as cross-encoders,
SPLADE, and ColBERT.

.. contents:: On this page
   :local:
   :depth: 2

Models
------

Wrappers around HuggingFace model classes. These configurations define which
pre-trained model to use and how it should be loaded.

.. autoxpmconfig:: xpmir.text.huggingface.base.HFMaskedLanguageModel
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModel

Init tasks
----------

Tasks that handle model weight loading at experiment submit time (from a
HuggingFace model ID or a local checkpoint).

.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelInitFromID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFFromCheckpoint

Tokenizers
----------

HuggingFace tokenizer wrappers, with variants for different output formats
(token IDs, strings, lists).

.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizerBase

.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFListTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFStringTokenizer
.. autoxpmconfig:: xpmir.text.huggingface.tokenizers.HFTokenizerAdapter

Encoders
--------

Encoders that produce text representations from HuggingFace models. These
implement the :class:`~xpmir.text.encoders.TokensEncoder` interface.

.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFEncoderBase

.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFTokensEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.HFCLSEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.OneHotHuggingFaceEncoder
.. autoxpmconfig:: xpmir.text.huggingface.encoders.SentenceTransformerTextEncoder

Training hooks
--------------

Hooks that modify encoder behaviour during training (e.g. selecting
intermediate layers).

.. autoxpmconfig:: xpmir.text.huggingface.encoders.LayerSelector
