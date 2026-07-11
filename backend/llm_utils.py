import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from backend.config import DEFAULT_MODEL_NAME

# Configure Google GenAI
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

async def generate_response(query: str, chunks: List[Dict]) -> str:
    """
    Generate a polished response based on retrieved paper chunks using Gemini (Non-blocking).
    """
    # Check environment again in case it was set dynamically after import
    active_key = os.environ.get("GEMINI_API_KEY")
    if not active_key:
        return "Error: GEMINI_API_KEY environment variable is not set. Please set it in your .env file."

    genai.configure(api_key=active_key)

    if not chunks:
        return "No relevant context found to answer the query."

    # Format the context chunks
    context_str = ""
    for chunk in chunks:
        paper_id = chunk.get("paper_id", "Unknown_Paper")
        clean_paper_name = paper_id.replace("_", " ")
        content = chunk.get("content", chunk.get("chunk_preview", ""))
        context_str += (
            f"--- Paper ID: {paper_id} ---\n"
            f"Paper Name: {clean_paper_name}\n"
            f"{content}\n\n"
        )

    # Construct the prompt
    prompt = f"""You are ResearchLens AI, a professional research assistant.
Your task is to answer the user's research query.

Strict Grounding & Completeness Instructions:
1. Answer the query using ONLY the provided paper chunks as context.
2. If a paper is not explicitly present in the "Context Chunks" below, you are strictly FORBIDDEN from discussing, summarizing, or referencing it in your response. Do not use your own pre-trained knowledge to fill in gaps for papers missing from the context.
3. If the context describes a multi-step process, methodology stages, or sequential steps (e.g. training pipelines, architectures), you MUST identify and include all distinct stages or steps mentioned in the context in your response. Do not omit the initial or final steps.
4. If the context does not contain enough information to answer the query for a specific paper, state: "Context insufficient to answer for [Paper Name]."
5. Always cite your sources by placing the exact Paper ID in square brackets at the end of the sentence or statement (e.g., [Attention_is_all_you_need]). Do not use space-separated names for citations; use the exact Paper ID with underscores.

Research Query: {query}

Context Chunks:
{context_str}

Answer:"""

    try:
        # Enforce deterministic, high-fidelity factual synthesis by setting temperature=0.0
        model = genai.GenerativeModel(
            DEFAULT_MODEL_NAME,
            generation_config={"temperature": 0.0}
        )
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response from Gemini API: {str(e)}"
