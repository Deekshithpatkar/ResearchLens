from backend.text_chunker import chunk_text


def test_section_chunking_keeps_section_headings():
    text = """
    Abstract
    This paper introduces a new method for document retrieval.

    Introduction
    The problem is that generic embeddings may miss section-specific context.

    Method
    We split papers into sections before chunking them for retrieval.

    Conclusion
    This improves relevance for objective and contribution questions.
    """

    chunks = chunk_text(text, chunk_size=12)

    assert len(chunks) >= 3
    assert any("Abstract" in chunk for chunk in chunks)
    assert any("Introduction" in chunk for chunk in chunks)
    assert any("Method" in chunk for chunk in chunks)
