import torch
from xpmir.neural.sentence_transformers import st_cross_scorer
from xpmir.letor.records import PointwiseItems


def test_st_cross_encoder_basic():
    model_id = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    scorer, init_tasks = st_cross_scorer(model_id=model_id, max_length=16)
    scorer = scorer.instance()

    # Run initialization tasks
    for task in init_tasks:
        task.instance().execute()

    # Check if ST model is loaded
    assert hasattr(scorer, "st_model")

    # Create sample data (repeating query for each document)
    records = PointwiseItems.from_texts(
        topics=["Red Planet", "Red Planet"],
        documents=["Mars is the Red Planet", "Venus is hot"],
        relevances=[1.0, 0.0],
    )

    # Test forward without pre-tokenization
    # Using __call__ which calls forward
    scores = scorer(records)

    print(scores)
    assert isinstance(scores, torch.Tensor)
    # Output can be (2, 1) or (2,) depending on the model
    scores = scores.view(-1)
    assert scores.shape[0] == 2

    # Test batch_tokenize
    tokenized = scorer.batch_tokenize(records)
    assert tokenized.ids is not None
    assert tokenized.ids.shape[0] == 2

    # Test forward with pre-tokenization
    scores_tokenized = scorer(records, tokenized=tokenized).view(-1)
    # The scores should be identical because they use the same underlying model and parameters
    assert torch.allclose(scores, scores_tokenized, atol=1e-5)


def test_st_cross_encoder_templates():
    model_id = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

    # Using templates similar to the user example
    query_template = "Query: {query}"
    doc_template = "Document: {document}"

    scorer, init_tasks = st_cross_scorer(
        model_id=model_id,
        max_length=32,
        query_template=query_template,
        document_template=doc_template,
    )
    scorer = scorer.instance()

    for task in init_tasks:
        task.instance().execute()

    records = PointwiseItems.from_texts(
        topics=["Red Planet"], documents=["Mars is the Red Planet"], relevances=[1.0]
    )

    # This should apply templates internally
    scores = scorer(records)
    assert scores.view(-1).shape[0] == 1


if __name__ == "__main__":
    test_st_cross_encoder_basic()
    test_st_cross_encoder_templates()
    print("Tests passed!")
