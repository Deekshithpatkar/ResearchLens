import re
import numpy as np
from typing import List, Dict
from backend.config import collection
from backend.text_chunker import preprocess_text
from backend.embedding_utils_simple import generate_embeddings
from backend.metadata_utils import load_metadata

def build_summary_text(text: str, max_words: int = 180) -> str:
    """Build a concise summary-like document from the abstract and opening content of a paper."""
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""

    abstract_match = re.search(r"abstract\s*(.*?)(?=\s*(?:\d+\s+)?introduction\b)", cleaned, re.I | re.S)
    if abstract_match:
        candidate = abstract_match.group(1).strip()
        if len(candidate.split()) >= 20:
            return candidate[:2000]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    useful_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        if any(token in lower for token in ["google", "brain", "research", "mailto", "www."]):
            continue
        useful_sentences.append(sentence)
        if len(" ".join(useful_sentences).split()) >= max_words:
            break

    summary = " ".join(useful_sentences[:4]).strip()
    if len(summary.split()) < 20:
        summary = cleaned[:1200]
    return summary[:2000]

def expand_query(query: str) -> str:
    """Expand a user query with useful synonyms for paper QA."""
    lowered = query.lower()
    expanded_terms = [query]
    if any(word in lowered for word in ["objective", "objectives", "goal", "goals"]):
        expanded_terms.append("purpose contribution aim")
    if any(word in lowered for word in ["problem", "solve", "solving"]):
        expanded_terms.append("challenge motivation issue")
    if any(word in lowered for word in ["method", "methodology", "approach"]):
        expanded_terms.append("model framework technique")
    if any(word in lowered for word in ["contribution", "contributions", "contribute"]):
        expanded_terms.append("novelty key contribution")
    return " ".join(expanded_terms)

def keyword_overlap_score(query: str, text: str) -> float:
    """Simple token overlap score for hybrid reranking."""
    if not query or not text:
        return 0.0

    stopwords = {"the", "a", "an", "is", "are", "what", "how", "why", "this", "that", "of", "to", "and", "for", "in", "on", "with", "from", "paper", "thesis"}

    q_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in stopwords and len(token) > 2}
    t_tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords and len(token) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / max(1, len(q_tokens))

def execute_semantic_search(query: str, paper_id: str = None, top_k: int = 5) -> Dict:
    """Core search logic, raises ValueError / KeyError instead of HTTPException"""
    if not query or len(query.strip()) == 0:
        raise ValueError("Query cannot be empty")

    cleaned_query = preprocess_text(query)
    expanded_query = expand_query(cleaned_query)
    query_vectors = generate_embeddings([cleaned_query, expanded_query], for_query=True)
    query_embedding = np.mean(query_vectors, axis=0)

    metadata = load_metadata()
    papers_to_search = {paper_id: metadata["papers"][paper_id]} if paper_id else metadata.get("papers", {})

    if paper_id and paper_id not in papers_to_search:
        raise KeyError(f"Paper '{paper_id}' not found")

    query_kwargs = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": max(top_k * 5, 12),
        "include": ["metadatas", "documents", "distances"],
    }
    if paper_id:
        query_kwargs["where"] = {"paper_id": paper_id}

    query_results = collection.query(**query_kwargs)

    if not query_results["ids"] or len(query_results["ids"][0]) == 0:
        return {
            "query": query,
            "results": [],
            "message": "No chunks found"
        }

    ranked_results = []
    for doc, meta, distance in zip(
        query_results["documents"][0],
        query_results["metadatas"][0],
        query_results["distances"][0],
    ):
        cleaned_doc = preprocess_text(doc)
        text = cleaned_doc or doc
        lowered = text.lower()
        boost = 0.0

        is_overview_query = any(word in query.lower() for word in ["summary", "objective", "goal", "overview", "abstract", "propose", "introduce", "contribution", "what is the paper about", "what does this paper do"])
        if meta.get("chunk_type") == "summary" and is_overview_query:
            boost += 0.12
        if any(word in lowered for word in ["abstract", "objective", "goal", "purpose", "problem", "contribution", "introduc", "method", "results"]):
            boost += 0.06
        if any(word in lowered for word in ["dataset", "model", "architecture", "attention", "transformer"]):
            boost += 0.01
        if len(text.split()) < 40:
            boost -= 0.02
        if len(text.split()) > 80:
            boost += 0.02

        keyword_score = keyword_overlap_score(cleaned_query, text)
        similarity = float(1.0 / (1.0 + distance)) + boost + (0.05 * keyword_score)
        ranked_results.append({
            "paper_id": meta.get("paper_id"),
            "chunk_index": meta.get("chunk_index"),
            "chunk_type": meta.get("chunk_type", "chunk"),
            "chunk_preview": text[:220] + "..." if len(text) > 220 else text,
            "content": doc,
            "similarity_score": similarity,
        })

    ranked_results = sorted(ranked_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

    return {
        "query": query,
        "num_results": len(ranked_results),
        "results": ranked_results
    }
