"""Unit tests for the ColBERT max-sim scoring operator.

These tests exercise the pure tensor code in
:class:`xpmir.neural.colbert.ColBERTEncoder` without requiring an actual HF
encoder or network access. They build fake
:class:`~xpmir.text.encoders.TokensRepresentationOutput` values and invoke the
unbound ``_max_sim`` method directly.

A separate integration test (:func:`test_pylate_parity`) checks that our
encoder produces the same token embeddings and MaxSim scores as PyLate's
reference ColBERT implementation when both models share the same backbone and
projection weights.
"""

import types

import pytest
import torch

from datamaestro_ir.data import SimpleTextItem

from xpmir.neural.colbert import ColBERTEncoder
from xpmir.text.encoders import (
    TokenizedTextEncoder,
    TokensRepresentationOutput,
)
from xpmir.text.tokenizers import TokenizedTexts
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.huggingface.base import HFConfigID, HFModel
from xpmir.text.huggingface.encoders import HFTokensEncoder
from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter

from xpmir.test import skip_if_ci


def _make_output(value: torch.Tensor, mask: torch.Tensor) -> TokensRepresentationOutput:
    tokenized = TokenizedTexts(
        None,
        torch.zeros(value.shape[:2], dtype=torch.long),
        [int(m.sum()) for m in mask],
        mask,
        None,
    )
    return TokensRepresentationOutput(value=value, tokenized=tokenized)


def _dummy():
    """A bare namespace that exposes ``_token_mask`` for unbound calls."""
    dummy = types.SimpleNamespace()
    dummy._token_mask = ColBERTEncoder._token_mask
    return dummy


def test_max_sim_all_pairs_matches_manual():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    d = torch.nn.functional.normalize(torch.randn(3, 5, 8), dim=-1)
    q_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    d_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool
    )
    q_out = _make_output(q, q_mask)
    d_out = _make_output(d, d_mask)

    scores = ColBERTEncoder._max_sim(_dummy(), q_out, d_out, all_pairs=True)
    assert scores.shape == (2, 3)

    # Reference computation for every (query, doc) pair.
    for i in range(2):
        for j in range(3):
            sim = q[i] @ d[j].T
            sim = sim.masked_fill(~d_mask[j].unsqueeze(0), float("-inf"))
            per_token_max = sim.max(dim=-1).values
            expected = (per_token_max * q_mask[i].float()).sum()
            assert torch.allclose(scores[i, j], expected, atol=1e-5)


def test_max_sim_pairs_matches_diagonal():
    torch.manual_seed(1)
    q = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    d = torch.nn.functional.normalize(torch.randn(2, 5, 8), dim=-1)
    q_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    d_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    q_out = _make_output(q, q_mask)
    d_out = _make_output(d, d_mask)

    pair_scores = ColBERTEncoder._max_sim(_dummy(), q_out, d_out, all_pairs=False)
    all_scores = ColBERTEncoder._max_sim(_dummy(), q_out, d_out, all_pairs=True)
    assert pair_scores.shape == (2,)
    assert torch.allclose(pair_scores, torch.diagonal(all_scores), atol=1e-5)


# ---------------------------------------------------------------- PyLate parity


_PYLATE_HF_ID = "hf-internal-testing/tiny-random-BertForMaskedLM"
_PYLATE_EMBED_DIM = 8
_PYLATE_QUERY_LEN = 8
_PYLATE_DOC_LEN = 16


def _records(texts):
    return [{"text_item": SimpleTextItem(t)} for t in texts]


def _build_pylate(seed: int):
    """Build a PyLate ColBERT configured to match the original ColBERT
    semantics implemented by :class:`ColBERTEncoder` (no [Q]/[D] prefixes,
    no skiplist, mask-augmented queries that attend to expansion tokens)."""
    from pylate import models

    model = models.ColBERT(
        _PYLATE_HF_ID,
        embedding_size=_PYLATE_EMBED_DIM,
        bias=False,
        query_prefix="",
        document_prefix="",
        do_query_expansion=True,
        attend_to_expansion_tokens=True,
        skiplist_words=[],
        query_length=_PYLATE_QUERY_LEN,
        document_length=_PYLATE_DOC_LEN,
    )
    # Re-seed the random projection for reproducibility.
    torch.manual_seed(seed)
    torch.nn.init.xavier_uniform_(model[1].linear.weight)
    model.to("cpu")
    model.eval()
    return model


def _build_xpmir(pylate_model) -> ColBERTEncoder:
    """Build an xpmir ColBERTEncoder sharing the BERT weights and projection
    of ``pylate_model``."""
    encoder = TokenizedTextEncoder.C(
        tokenizer=HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=_PYLATE_HF_ID),
            converter=TopicTextConverter.C(),
        ),
        encoder=HFTokensEncoder.C(
            model=HFModel.C(config=HFConfigID.C(hf_id=_PYLATE_HF_ID))
        ),
    )
    colbert = ColBERTEncoder.C(
        encoder=encoder,
        dim=_PYLATE_EMBED_DIM,
        query_maxlen=_PYLATE_QUERY_LEN,
        doc_maxlen=_PYLATE_DOC_LEN,
        query_augmentation=True,
    ).instance()
    colbert.initialize()

    # Share the BERT backbone and projection with the PyLate model so both
    # models produce identical contextual embeddings.
    bert = pylate_model[0].auto_model
    colbert.encoder.encoder.model.model.load_state_dict(bert.state_dict())
    colbert._projection.weight.data.copy_(pylate_model[1].linear.weight.data)
    colbert.eval()
    return colbert


@skip_if_ci
def test_pylate_parity_query_embeddings():
    """Per-token query embeddings (with [MASK] augmentation) should match
    PyLate's reference ColBERT implementation."""
    pytest.importorskip("pylate.models", exc_type=ImportError)

    pylate = _build_pylate(seed=0)
    xpmir = _build_xpmir(pylate)

    queries = ["what is colbert"]

    with torch.no_grad():
        py_emb = pylate.encode(
            queries,
            is_query=True,
            padding=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # PyLate returns a list of (1, Lq, dim) tensors when batched per item.
        py_q = torch.stack([e.squeeze(0) for e in py_emb])  # (B, Lq, dim)

        xp_out = xpmir.encode_queries(_records(queries))
        xp_q = xp_out.value.detach()

    assert xp_q.shape == py_q.shape, (xp_q.shape, py_q.shape)
    torch.testing.assert_close(xp_q, py_q, atol=1e-5, rtol=1e-4)


@skip_if_ci
def test_pylate_parity_maxsim_scores():
    """The MaxSim score matrix should match PyLate's ``colbert_scores`` when
    both models share weights."""
    pytest.importorskip("pylate.models", exc_type=ImportError)
    pytest.importorskip("pylate.scores", exc_type=ImportError)
    from pylate import scores as pylate_scores

    pylate = _build_pylate(seed=1)
    xpmir = _build_xpmir(pylate)

    queries = ["what is colbert", "late interaction"]
    documents = [
        "colbert is a late interaction model",
        "completely unrelated text about cats",
    ]

    with torch.no_grad():
        py_q = torch.stack(
            [
                e.squeeze(0)
                for e in pylate.encode(
                    queries,
                    is_query=True,
                    padding=True,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            ]
        )
        py_d_list = pylate.encode(
            documents,
            is_query=False,
            padding=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Pad PyLate's per-doc tensors to the longest length to build a mask.
        py_d_squeezed = [e.squeeze(0) for e in py_d_list]
        max_len = max(t.shape[0] for t in py_d_squeezed)
        d_mask = torch.zeros(len(py_d_squeezed), max_len, dtype=torch.bool)
        py_d = torch.zeros(len(py_d_squeezed), max_len, _PYLATE_EMBED_DIM)
        for i, t in enumerate(py_d_squeezed):
            py_d[i, : t.shape[0]] = t
            d_mask[i, : t.shape[0]] = True

        py_score = pylate_scores.colbert_scores(py_q, py_d, documents_mask=d_mask)

        xp_q_out = xpmir.encode_queries(_records(queries))
        xp_d_out = xpmir.encode_documents(_records(documents))
        xp_score = xpmir.score_product(xp_q_out, xp_d_out).detach()

    assert xp_score.shape == py_score.shape, (xp_score.shape, py_score.shape)
    torch.testing.assert_close(xp_score, py_score, atol=1e-4, rtol=1e-4)
