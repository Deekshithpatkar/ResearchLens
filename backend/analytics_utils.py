import json
import os
import re
import asyncio
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

from backend.config import PROFILES_DIR, SCHEMA_VERSION, collection, DEFAULT_MODEL_NAME
from backend.metadata_utils import load_metadata
from backend.profile_utils import load_profile, extract_paper_profile, save_profile

# Ensure environment variables are loaded
load_dotenv()

# Configure Google GenAI
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Global tracker to prevent asyncio task garbage collection
_running_tasks = set()

# Allowed schema keys for field projection
ALLOWED_SCHEMA_KEYS = [
    "title", "authors", "publication_year", "research_problem", "objective",
    "methodology", "model_architecture", "datasets", "evaluation_metrics",
    "main_results", "contributions", "author_stated_limitations",
    "inferred_limitations", "future_work", "keywords", "research_topics"
]

def determine_relevant_fields(query: str) -> List[str]:
    """Use Gemini to dynamically classify which schema keys are relevant to the user query."""
    if not api_key:
        return ALLOWED_SCHEMA_KEYS

    prompt = f"""You are a query router for a research paper search engine.
Your task is to classify which structured schema fields are relevant to answer the user's query.

User Query: "{query}"

Available Schema Fields:
{json.dumps(ALLOWED_SCHEMA_KEYS, indent=2)}

Strict Instructions:
1. Return ONLY a valid JSON list of strings (e.g., ["methodology", "datasets"]).
2. Select only the fields that are directly needed to answer the query. Do not select fields that are irrelevant.
3. If the query is broad or you are unsure, select all fields.

JSON Output:"""

    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean potential markdown wrappers
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
            
        fields = json.loads(text)
        if isinstance(fields, list):
            # Validate fields are allowed
            valid_fields = [f for f in fields if f in ALLOWED_SCHEMA_KEYS]
            return valid_fields if valid_fields else ALLOWED_SCHEMA_KEYS
    except Exception as e:
        print(f"Error determining relevant fields (defaulting to all): {e}")
        
    return ALLOWED_SCHEMA_KEYS

async def execute_global_analytics(query: str, paper_ids: List[str] = None) -> Dict:
    """
    Execute global query analysis across a subset or all papers in the database.
    Performs lazy schema migrations, field routing, and output coverage checking.
    """
    metadata = load_metadata()
    all_paper_ids = list(metadata.get("papers", {}).keys())

    # Resolve target paper_ids
    if not paper_ids or "all" in paper_ids:
        target_ids = all_paper_ids
    else:
        target_ids = [pid for pid in paper_ids if pid in all_paper_ids]

    completed_profiles = []
    warnings = []

    # Process and load profiles
    for pid in target_ids:
        profile = load_profile(pid)
        paper_info = metadata["papers"][pid]
        filename = paper_info.get("filename", f"{pid}.pdf")

        if not profile:
            # Profile doesn't exist, trigger background generation
            warnings.append(f"Paper '{pid}' profile was missing. Triggering extraction task.")
            t = asyncio.create_task(extract_paper_profile(pid, filename))
            _running_tasks.add(t)
            t.add_done_callback(_running_tasks.discard)
            continue

        status = profile.get("profile_status", "failed")

        if status == "processing":
            warnings.append(f"Paper '{pid}' is currently processing and was excluded.")
            continue
        elif status == "failed":
            warnings.append(f"Paper '{pid}' failed extraction previously: {profile.get('error') or 'Unknown error'}")
            continue
        elif status == "migrating":
            # In progress migration, load current old data
            completed_profiles.append(profile)
            continue
        elif status == "completed":
            # Check for lazy schema migration
            if profile.get("schema_version") != SCHEMA_VERSION:
                warnings.append(f"Paper '{pid}' has an outdated schema. Triggering lazy migration in the background.")
                # Mark as migrating to prevent duplicate queues
                profile["profile_status"] = "migrating"
                save_profile(pid, profile)
                # Launch background update with force_reextract=True to bypass migrations guard
                t = asyncio.create_task(extract_paper_profile(pid, filename, force_reextract=True))
                _running_tasks.add(t)
                t.add_done_callback(_running_tasks.discard)
            
            completed_profiles.append(profile)

    if not completed_profiles:
        return {
            "answer": "No completed paper profiles are currently available to perform this analysis.",
            "warnings": warnings,
            "fields_used": []
        }

    # Dynamic Field Routing
    relevant_fields = determine_relevant_fields(query)
    # Ensure paper_id and title are always included for identification
    projected_fields = list(set(["paper_id", "title"] + relevant_fields))

    # Project (filter) profile data
    projected_profiles = []
    for prof in completed_profiles:
        proj = {k: prof.get(k) for k in projected_fields if k in prof}
        projected_profiles.append(proj)

    # Synthesize Answer using Map-Reduce Scaling Logic (Fix #5)
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL_NAME)
        batch_size = 5
        
        if len(projected_profiles) <= 6:
            # Single-shot synthesis (Direct Reduce)
            prompt = f"""You are ResearchLens AI, a senior academic research synthesist.
Your task is to analyze the user's global query using the provided structured profiles of the selected research papers.

User Query: "{query}"

Selected Paper Profiles (Projected Fields):
{json.dumps(projected_profiles, indent=2, ensure_ascii=False)}

Instructions:
1. Synthesize a comprehensive, clear, and well-structured response.
2. Represent comparisons or summaries in a structured Markdown Table.
3. Every paper profile provided MUST be represented in your answer. Do not ignore or drop any papers. If a field is empty, write "Not explicitly stated".
4. For each claim, cite the source paper using its exact paper_id in brackets (e.g., [Attention_is_all_you_need]).

Answer:"""
            response = await model.generate_content_async(prompt)
            answer = response.text
        else:
            # Multi-batch Map-Reduce
            intermediate_summaries = []
            
            # 1. Map Step: Process in batches of 5
            for i in range(0, len(projected_profiles), batch_size):
                batch = projected_profiles[i:i + batch_size]
                map_prompt = f"""You are ResearchLens AI.
Analyze the following subset of paper profiles and generate a detailed intermediate comparison addressing the query: "{query}".
Make sure every paper in this subset is represented and cited using its exact paper_id in brackets. If a field is empty, write "Not explicitly stated".

Subset Paper Profiles:
{json.dumps(batch, indent=2, ensure_ascii=False)}

Intermediate Synthesis:"""
                res = await model.generate_content_async(map_prompt)
                intermediate_summaries.append(res.text)
                
            # 2. Reduce Step: Synthesize intermediate summaries into final answer
            reduce_prompt = f"""You are ResearchLens AI, a professional research assistant.
Synthesize the following intermediate paper comparison summaries into a single, cohesive, and comprehensive final response to answer the query: "{query}".

Instructions:
1. Represent the comparison in a single, structured Markdown Table.
2. Every paper ID from the summaries MUST have its own row in the table. Do not drop any papers.
3. For each claim, cite using the exact paper_id in brackets (e.g., [BERT]).

Intermediate Comparison Summaries:
{"\n\n---\n\n".join(intermediate_summaries)}

Final Answer:"""
            res = await model.generate_content_async(reduce_prompt)
            answer = res.text

        # Output Coverage Check Loop
        missing_pids = []
        for prof in completed_profiles:
            pid = prof["paper_id"]
            # Check if paper_id is mentioned in the output answer (normalized for spaces/underscores/case)
            normalized_pid = pid.replace("_", " ").lower()
            normalized_answer = answer.lower()
            if pid not in answer and normalized_pid not in normalized_answer and pid.lower() not in normalized_answer:
                missing_pids.append(pid)

        if missing_pids:
            print(f"Coverage Check Alert: Gemini omitted {len(missing_pids)} papers: {missing_pids}. Triggering supplementary synthesis.")
            
            # Format projected profiles of missing papers
            missing_profiles = [p for p in projected_profiles if p["paper_id"] in missing_pids]
            
            supplement_prompt = f"""You are ResearchLens AI. In your previous output, you omitted the following papers.
Provide a supplementary structured response (Markdown table rows or sections) specifically for these omitted papers to answer the query: "{query}".

Omitted Paper Profiles:
{json.dumps(missing_profiles, indent=2, ensure_ascii=False)}

Output ONLY the additional rows/content for these papers so they can be merged directly into the main output. Do not repeat the papers you already answered.

Supplementary Answer:"""
            
            supp_response = await model.generate_content_async(supplement_prompt)
            supplement_text = supp_response.text
            
            # Simple stitching: Append the supplement at the end or attempt to merge
            answer += f"\n\n### Supplementary Analysis (Omitted Papers)\n{supplement_text}"

        return {
            "answer": answer,
            "warnings": warnings,
            "fields_used": relevant_fields
        }

    except Exception as e:
        return {
            "answer": f"Error performing synthesis analysis: {str(e)}",
            "warnings": warnings,
            "fields_used": relevant_fields
        }

