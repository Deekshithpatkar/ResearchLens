import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from typing import Optional
from backend.config import CHUNKS_DIR, PROFILES_DIR, SCHEMA_VERSION, get_extraction_semaphore, DEFAULT_MODEL_NAME, get_user_profiles_dir, get_user_chunks_dir

# Ensure environment variables are loaded
load_dotenv()

# Configure Google GenAI
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def save_profile(paper_id: str, profile_data: dict, user_id: Optional[str] = None):
    """Save paper profile locally as JSON, optionally scoped by user."""
    if user_id:
        profile_path = get_user_profiles_dir(user_id) / f"{paper_id}.json"
    else:
        profile_path = PROFILES_DIR / f"{paper_id}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)

def load_profile(paper_id: str, user_id: Optional[str] = None) -> dict:
    """Load paper profile locally, optionally scoped by user."""
    if user_id:
        profile_path = get_user_profiles_dir(user_id) / f"{paper_id}.json"
    else:
        profile_path = PROFILES_DIR / f"{paper_id}.json"
    if not profile_path.exists():
        return None
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_json_response(raw_text: str) -> str:
    """Clean markdown json wrappers and general parsing anomalies from LLM output."""
    cleaned = raw_text.strip()
    # Strip markdown code blocks
    if cleaned.startswith("```"):
        # Remove starting ```json or ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        # Remove ending ```
        cleaned = re.sub(r"\s*```$", "", cleaned)
    
    cleaned = cleaned.strip()
    return cleaned

def parse_with_repair(raw_text: str) -> dict:
    """Parse JSON string with simple repair strategies for trailing commas or malformed fences."""
    cleaned = clean_json_response(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Simple regex repair: remove trailing commas before closing braces/brackets
        repaired = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Re-raise original error to trigger secondary retry
            raise e

async def extract_paper_profile(paper_id: str, file_name: str, force_reextract: bool = False, user_id: Optional[str] = None):
    """
    Background worker task to extract structured paper profiles from raw chunks.
    Throttled by a semaphore to prevent API rate limits.
    """
    semaphore = get_extraction_semaphore()
    async with semaphore:
        # Check current status; avoid duplicate background extractions
        current = load_profile(paper_id, user_id=user_id)
        if not force_reextract and current and current.get("profile_status") in ("processing", "migrating"):
            print(f"Skipping redundant profile extraction for {paper_id} (status: {current.get('profile_status')})")
            return

        # Set status to processing
        save_profile(paper_id, {
            "paper_id": paper_id,
            "schema_version": SCHEMA_VERSION,
            "profile_status": "processing",
            "error": None,
            "filename": file_name,
            "uploaded_at": datetime.now().isoformat()
        }, user_id=user_id)

        if user_id:
            chunks_file = get_user_chunks_dir(user_id) / f"{paper_id}.json"
        else:
            chunks_file = CHUNKS_DIR / f"{paper_id}.json"

        if not chunks_file.exists():
            save_profile(paper_id, {
                "paper_id": paper_id,
                "schema_version": SCHEMA_VERSION,
                "profile_status": "failed",
                "filename": file_name,
                "error": "Chunks file not found on disk"
            }, user_id=user_id)
            return

        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                raise ValueError("No text chunks found for this paper")

            # Prepare numbered chunks context for citation extraction
            # For papers > 100 chunks, select the first 15, last 15, and fill remaining slots prioritizing Methodology/Results/Limitations sections
            max_chunks = 100
            if len(chunks) > max_chunks:
                start_indices = list(range(15))
                end_indices = list(range(len(chunks) - 15, len(chunks)))
                middle_indices = list(range(15, len(chunks) - 15))
                
                # Identify chunks belonging to key sections
                key_section_indices = []
                other_middle_indices = []
                key_patterns = ["method", "approach", "model", "architecture", "experiment", "result", "discussion", "limitation", "future"]
                
                for idx in middle_indices:
                    text = chunks[idx].lower()
                    colon_idx = text.find(":")
                    prefix = text[:colon_idx] if colon_idx != -1 else ""
                    if any(pat in prefix for pat in key_patterns):
                        key_section_indices.append(idx)
                    else:
                        other_middle_indices.append(idx)
                
                # Fill remaining slots (70 slots)
                remaining_slots = max_chunks - (len(start_indices) + len(end_indices))
                selected_middle = key_section_indices[:remaining_slots]
                
                slots_left = remaining_slots - len(selected_middle)
                if slots_left > 0 and other_middle_indices:
                    step = max(1, len(other_middle_indices) // slots_left)
                    selected_other = [other_middle_indices[i] for i in range(0, len(other_middle_indices), step)][:slots_left]
                    selected_middle.extend(selected_other)
                
                indices = sorted(list(set(start_indices + selected_middle + end_indices)))
                selected_chunks = [chunks[idx] for idx in indices]
            else:
                selected_chunks = chunks
                indices = list(range(len(chunks)))

            context_lines = []
            for idx, text in zip(indices, selected_chunks):
                context_lines.append(f"[chunk_{idx}]\n{text}\n")
            context_str = "\n".join(context_lines)

            # Strict structured prompt instructing Gemini
            prompt = f"""You are a scholarly paper analysis assistant.
Analyze the following academic paper text (divided into labeled chunks [chunk_0], [chunk_1], etc.) and extract a detailed structured profile.

Strict Instructions:
1. Return ONLY a valid JSON object matching the schema below.
2. For each extracted item in the fields (like objectives, methodology, limitations), identify the EXACT verbatim quote and map it in the "evidence" dictionary.
3. Every evidence item must have:
   - "text": The exact quote from the paper.
   - "section": The section name (e.g., "Introduction", "Limitations", "Abstract").
   - "chunk_id": The chunk label where it was found (e.g., "chunk_5").

JSON Output Schema:
{{
  "title": "Clean full title of the paper",
  "authors": ["Author Name 1", "Author Name 2"],
  "publication_year": 2017 (integer, null if unknown),
  "research_problem": "Detailed statement of the problem the paper addresses",
  "objective": "Main objective/goal of the paper",
  "methodology": "Overview of the methodologies or algorithms used",
  "model_architecture": "Details of the neural net or model architectures, null if not applicable",
  "datasets": ["Dataset Name 1", "Dataset Name 2"],
  "evaluation_metrics": ["Metric 1", "Metric 2"],
  "main_results": ["Key Result 1", "Key Result 2"],
  "contributions": ["Contribution 1", "Contribution 2"],
  "author_stated_limitations": ["Limitation explicitly written by the authors"],
  "inferred_limitations": ["Limitation that can be logically inferred from the experiments or scope"],
  "future_work": ["Suggested future directions stated by the authors"],
  "keywords": ["Keyword 1", "Keyword 2"],
  "research_topics": ["Broad topic/field, e.g., NLP, Computer Vision, Optimization"],
  "evidence": {{
     "objective": [
        {{ "text": "exact quote...", "section": "Introduction", "chunk_id": "chunk_2" }}
     ],
     "methodology": [
        {{ "text": "exact quote...", "section": "Methodology", "chunk_id": "chunk_8" }}
     ],
     "author_stated_limitations": [
        {{ "text": "exact quote...", "section": "Conclusion", "chunk_id": "chunk_45" }}
     ],
     "future_work": [
        {{ "text": "exact quote...", "section": "Future Work", "chunk_id": "chunk_46" }}
     ]
  }}
}}

Paper Text:
{context_str}

JSON Output:"""

            # Call Gemini
            model = genai.GenerativeModel(DEFAULT_MODEL_NAME)
            response = await model.generate_content_async(prompt)
            raw_text = response.text

            try:
                profile_data = parse_with_repair(raw_text)
            except Exception as parse_err:
                print(f"JSON parsing failed for {paper_id}, attempting correction call. Error: {parse_err}")
                # Retry once with correction prompt
                retry_prompt = f"The following text is not valid JSON. Repair and output only valid JSON according to the schema:\n\n{raw_text}"
                response = await model.generate_content_async(retry_prompt)
                profile_data = parse_with_repair(response.text)

            # Validate evidence quotes against actual paper chunks to ensure absolute integrity (Fix #9)
            evidence = profile_data.get("evidence", {})
            if isinstance(evidence, dict):
                for field, quotes in list(evidence.items()):
                    if not isinstance(quotes, list):
                        continue
                    valid_quotes = []
                    for q in quotes:
                        if not isinstance(q, dict) or "text" not in q:
                            continue
                        
                        quote_text = q["text"].strip()
                        if not quote_text:
                            continue
                            
                        # Normalize text by stripping all non-alphanumeric characters for robust comparison against mangled PDF text
                        clean_quote = re.sub(r"[^a-zA-Z0-9]", "", quote_text).lower()
                        if not clean_quote:
                            continue
                        
                        # Parse chunk index
                        chunk_id = q.get("chunk_id", "")
                        chunk_idx = None
                        if chunk_id and chunk_id.startswith("chunk_"):
                            try:
                                chunk_idx = int(chunk_id.split("_")[1])
                            except (ValueError, IndexError):
                                pass
                        
                        verified = False
                        # Check designated chunk
                        if chunk_idx is not None and 0 <= chunk_idx < len(chunks):
                            clean_chunk = re.sub(r"[^a-zA-Z0-9]", "", chunks[chunk_idx]).lower()
                            if clean_quote in clean_chunk:
                                verified = True
                                
                        # Fallback: scan other chunks and correct the chunk_id if found
                        if not verified:
                            for idx, chunk_text in enumerate(chunks):
                                clean_chunk = re.sub(r"[^a-zA-Z0-9]", "", chunk_text).lower()
                                if clean_quote in clean_chunk:
                                    q["chunk_id"] = f"chunk_{idx}"
                                    verified = True
                                    break
                                    
                        if verified:
                            valid_quotes.append(q)
                        else:
                            print(f"Evidence quote verification failed for {paper_id} field '{field}': '{quote_text[:50]}...' not found. Omitted.")
                    
                    evidence[field] = valid_quotes
                profile_data["evidence"] = evidence

            # Populate metadata fields
            profile_data["paper_id"] = paper_id
            profile_data["schema_version"] = SCHEMA_VERSION
            profile_data["profile_status"] = "completed"
            profile_data["error"] = None
            profile_data["filename"] = file_name
            profile_data["uploaded_at"] = datetime.now().isoformat()

            save_profile(paper_id, profile_data, user_id=user_id)
            print(f"Successfully generated structured profile for {paper_id}")

        except Exception as e:
            print(f"Error extracting profile for {paper_id}: {str(e)}")
            save_profile(paper_id, {
                "paper_id": paper_id,
                "schema_version": SCHEMA_VERSION,
                "profile_status": "failed",
                "filename": file_name,
                "error": str(e),
                "uploaded_at": datetime.now().isoformat()
            }, user_id=user_id)
