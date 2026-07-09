import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google GenAI
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_response(query: str, chunks: List[Dict]) -> str:
    """
    Generate a polished response based on retrieved paper chunks using Gemini.
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
        paper_id = chunk.get("paper_id", "Unknown Paper")
        clean_paper_name = paper_id.replace("_", " ")
        content = chunk.get("content", chunk.get("chunk_preview", ""))
        context_str += f"--- Paper: {clean_paper_name} ---\n{content}\n\n"

    # Construct the prompt
    prompt = f"""You are ResearchLens AI, a professional research assistant.
Your task is to answer the user's research query using ONLY the provided paper chunks as context.
Synthesize a comprehensive, clear, and well-structured response.
If the context does not contain enough information to answer the query, state that clearly.
Always cite your sources by placing the name of the paper in square brackets at the end of the sentence or statement (e.g., [Attention is all you need]). Do not use source numbers or chunk numbers.

Research Query: {query}

Context Chunks:
{context_str}

Answer:"""

    try:
        # Use gemini-2.5-flash for general synthesis
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response from Gemini API: {str(e)}"
