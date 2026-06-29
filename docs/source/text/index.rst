Text representation
===================

The ``text`` module provides the building blocks for converting raw text into
numerical representations -- from tokenisation to contextual embeddings and
document-level vectors. These encoders are used by the neural models in
:doc:`/neural` and can be composed freely.

.. toctree::
   :maxdepth: 2

   huggingface


Base encoders
-------------

Abstract interfaces for encoders and token-level encoders.

.. autoxpmconfig:: xpmir.text.encoders.Encoder

.. autoxpmconfig:: xpmir.text.encoders.TokensEncoder
   :members: forward

Tokenizers
----------

Tokenizers split text into token sequences and manage the vocabulary mapping.

.. autoxpmconfig:: xpmir.text.tokenizers.Tokenizer
   :members: pad_sequences, batch_tokenize, pad_tokenid, tok2id, id2tok, lexicon_size

.. autoxpmconfig:: xpmir.text.tokenizers.TokenizerBase


Text encoders
-------------

Encoders that map a text string (or a pair of texts) to a dense representation.

.. autoxpmconfig:: xpmir.text.encoders.TextEncoderBase

.. autoxpmconfig:: xpmir.text.encoders.TextEncoder
   :members: forward

.. autoxpmconfig:: xpmir.text.encoders.DualTextEncoder
   :members: forward

.. autoxpmconfig:: xpmir.text.encoders.TripletTextEncoder

.. autoxpmconfig:: xpmir.text.encoders.TokenizedTextEncoderBase
   :members: forward

.. autoxpmconfig:: xpmir.text.encoders.TokenizedEncoder
   :members: forward

Tokenizer-based encoders
-------------------------

.. autoxpmconfig:: xpmir.text.encoders.TokenizedTextEncoder

Adapters
--------

Adapters transform or aggregate encoder outputs.

.. autoxpmconfig:: xpmir.text.adapters.MeanTextEncoder
.. autoxpmconfig:: xpmir.text.adapters.TopicTextConverter
