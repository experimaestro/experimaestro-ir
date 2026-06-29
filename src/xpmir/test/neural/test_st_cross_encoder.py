import torch
import pytest
from sentence_transformers import CrossEncoder
from xpmir.neural.sentence_transformers import st_cross_scorer
from xpmir.letor.records import PointwiseItems


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3-Reranker-0.6B",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "mixedbread-ai/mxbai-rerank-base-v2",
    ],
)
def test_st_cross_encoder_parity(model_id):
    """Verify parity between raw sentence-transformers and STCrossEncoder.
    Ensures that both direct and tokenized calls in xpmir yield the same scores as the raw ST implementation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # 1. Setup raw ST implementation
    model = CrossEncoder(model_id, device=str(device))

    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]

    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    ]

    pairs = [[query, doc] for query, doc in zip(queries, documents)]

    # Process one by one to avoid ST's internal batch sorting for exact parity
    raw_scores = []
    for pair in pairs:
        score = model.predict([pair], convert_to_tensor=True)
        raw_scores.append(score)
    raw_scores = torch.cat(raw_scores)

    # 2. Setup xpmir STCrossEncoder implementation
    scorer, init_tasks = st_cross_scorer(model_id=model_id)
    scorer = scorer.instance()
    scorer.to(device)

    # Run initialization tasks
    for task in init_tasks:
        task.instance().execute()

    # 3. Compare scores
    for i in range(len(queries)):
        records = PointwiseItems.from_texts(
            topics=[queries[i]], documents=[documents[i]]
        )

        # Direct call
        with torch.no_grad():
            score_dir = scorer(records).view(-1)

        # Pre-tokenized call
        tokenized = scorer.batch_tokenize(records)
        with torch.no_grad():
            score_tok = scorer(records, tokenized=tokenized).view(-1)

        # Assertions
        expected = raw_scores[i]
        # Use a slightly higher tolerance for BFloat16 vs Float32 if necessary
        # but Qwen3-Reranker-0.6B usually yields exact matches in sequential mode
        assert torch.allclose(score_dir.to(expected.dtype), expected, atol=1e-4), (
            f"Direct score mismatch at index {i}: expected {expected}, got {score_dir}"
        )
        assert torch.allclose(score_tok.to(expected.dtype), expected, atol=1e-4), (
            f"Tokenized score mismatch at index {i}: expected {expected}, got {score_tok}"
        )
