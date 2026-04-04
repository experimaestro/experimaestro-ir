SPLADE
======

Reproduction of

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant).
    2021. https://arxiv.org/abs/2205.04733

Using pre-trained models
------------------------

To create a SPLADE model from a pre-trained HuggingFace MLM checkpoint:

.. autofunction:: xpmir.neural.splade.splade_from_pretrained_hf

Factory functions
-----------------

.. autofunction:: xpmir.neural.splade.spladeV1

.. autofunction:: xpmir.neural.splade.spladeV2_max

.. autofunction:: xpmir.neural.splade.spladeV2_doc
