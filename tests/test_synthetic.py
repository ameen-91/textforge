import os
import pytest
import pandas as pd
from textforge.synthetic import SyntheticDataGeneration


def test_synthetic_classification():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set")
    labels = ["positive", "negative"]
    texts = [
        "I love this movie",
        "This product is awful",
        "The service was excellent",
        "I hated the food",
        "The experience was mediocre",
    ]
    data = pd.DataFrame(texts, columns=["text"])
    classifier = SyntheticDataGeneration(
        api_key=api_key, labels=labels, query="sentiment analysis", sync_client=True
    )
    result = classifier.run_sync(data)

    # Assert that a label column exists and contains expected values
    assert "label" in result.columns
    assert len(result) == len(texts)
    for label in result["label"]:
        assert label in labels
