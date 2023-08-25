Text Representation
-------------------

.. toctree::
   :maxdepth: 2

   huggingface
   wordvec


The `text` module groups classes and configurations that compute
a representation of text -- this includes word embeddings as well
as contextual word embeddings and document embeddings.

.. autoxpmconfig:: xpmir.text.encoders.Tokenizer
   :members: pad_sequences, batch_tokenize, pad_tokenid, tok2id, id2tok, lexicon_size

.. autoxpmconfig:: xpmir.text.encoders.TokensEncoder
   :members: forward

.. autoxpmconfig:: xpmir.text.encoders.Encoder
.. autoxpmconfig:: xpmir.text.encoders.MeanTextEncoder
.. autoxpmconfig:: xpmir.text.encoders.TripletTextEncoder

.. autoxpmconfig:: xpmir.text.encoders.TextEncoder
   :members: forward

.. autoxpmconfig:: xpmir.text.encoders.DualTextEncoder
   :members: forward
