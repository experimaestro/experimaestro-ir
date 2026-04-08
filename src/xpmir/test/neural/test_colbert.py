"""Unit tests for the ColBERT max-sim scoring operator.

These tests exercise the pure tensor code in
:class:`xpmir.neural.colbert.ColBERTEncoder` without requiring an actual HF
encoder or network access. They build fake
:class:`~xpmir.text.encoders.TokensRepresentationOutput` values and invoke the
unbound ``_max_sim`` method directly.
"""

import types

import torch

from xpmir.neural.colbert import ColBERTEncoder
from xpmir.text.encoders import TokensRepresentationOutput
from xpmir.text.tokenizers import TokenizedTexts


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
