import numpy as np

from backend.embedding_utils_simple import generate_embeddings


def test_generate_embeddings_capture_shared_keywords():
    texts = [
        "The objective of this paper is to improve retrieval quality for academic papers.",
        "This work studies the objective and goals of retrieval systems for scientific documents.",
    ]

    embeddings = generate_embeddings(texts)
    similarity = float(np.dot(embeddings[0], embeddings[1]))

    assert embeddings.shape[0] == 2
    assert similarity > 0.2
