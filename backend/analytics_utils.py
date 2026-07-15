import json
import os
import re
import asyncio
from typing import List, Dict, Optional
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

from backend.config import PROFILES_DIR, SCHEMA_VERSION, collection, DEFAULT_MODEL_NAME, EMBEDDINGS_DIR, get_user_embeddings_dir
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

async def execute_global_analytics(query: str, paper_ids: List[str] = None, user_id: str = None) -> Dict:
    """
    Execute global query analysis across a subset or all papers in the database.
    Performs lazy schema migrations, field routing, and output coverage checking.
    """
    metadata = load_metadata(user_id=user_id)
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
        profile = load_profile(pid, user_id=user_id)
        paper_info = metadata["papers"][pid]
        filename = paper_info.get("filename", f"{pid}.pdf")

        if not profile:
            # Profile doesn't exist, trigger background generation
            warnings.append(f"Paper '{pid}' profile was missing. Triggering extraction task.")
            t = asyncio.create_task(extract_paper_profile(pid, filename, user_id=user_id))
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
                save_profile(pid, profile, user_id=user_id)
                # Launch background update with force_reextract=True to bypass migrations guard
                t = asyncio.create_task(extract_paper_profile(pid, filename, force_reextract=True, user_id=user_id))
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

async def execute_cluster_analysis(user_id: str = None) -> Dict:
    """
    Perform mathematical connected-components clustering of all uploaded papers,
    and dynamically request Gemini to synthesize cluster names and descriptions.
    """
    metadata = load_metadata(user_id=user_id)
    all_paper_ids = list(metadata.get("papers", {}).keys())

    if not all_paper_ids:
        return {
            "clusters": [],
            "warnings": ["No papers uploaded to cluster."]
        }

    # 1. Load global embeddings
    paper_embeddings = {}
    for pid in all_paper_ids:
        if user_id:
            emb_path = get_user_embeddings_dir(user_id) / f"{pid}.npy"
        else:
            emb_path = EMBEDDINGS_DIR / f"{pid}.npy"
        if emb_path.exists():
            try:
                emb = np.load(emb_path)
                if len(emb) > 0:
                    paper_embeddings[pid] = emb[0]
            except Exception as e:
                print(f"Error loading embedding for {pid}: {e}")

    pids = list(paper_embeddings.keys())
    n = len(pids)
    
    if n == 0:
        return {
            "clusters": [],
            "warnings": ["No valid paper embeddings found for clustering."]
        }

    # 2. Compute similarity matrix
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v1 = paper_embeddings[pids[i]]
            v2 = paper_embeddings[pids[j]]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                similarity_matrix[i, j] = np.dot(v1, v2) / (norm1 * norm2)
            else:
                similarity_matrix[i, j] = 0.0

    # 3. Clustering (Connected components grouping)
    # Cosine similarity threshold: 0.65 represents highly related topics
    threshold = 0.65
    visited = set()
    raw_clusters = []

    for i in range(n):
        if pids[i] not in visited:
            component = []
            queue = [i]
            visited.add(pids[i])
            
            while queue:
                curr = queue.pop(0)
                component.append(pids[curr])
                
                for neighbor in range(n):
                    if pids[neighbor] not in visited:
                        if similarity_matrix[curr, neighbor] >= threshold:
                            visited.add(pids[neighbor])
                            queue.append(neighbor)
            raw_clusters.append(component)

    # 4. Generate AI labels for each cluster dynamically
    labeled_clusters = []
    warnings = []
    model = genai.GenerativeModel(DEFAULT_MODEL_NAME)

    for idx, cluster_pids in enumerate(raw_clusters):
        # Retrieve titles and objectives of all papers in this cluster
        paper_details = []
        for pid in cluster_pids:
            profile = load_profile(pid, user_id=user_id)
            if profile:
                paper_details.append({
                    "paper_id": pid,
                    "title": profile.get("title", pid),
                    "research_topics": profile.get("research_topics", []),
                    "objective": profile.get("objective", "Not stated")
                })
            else:
                paper_details.append({
                    "paper_id": pid,
                    "title": pid,
                    "research_topics": [],
                    "objective": "Not stated"
                })

        # Use Gemini to summarize the theme of this cluster
        prompt = f"""You are a senior academic research synthesist.
Analyze the following group of research papers and generate a descriptive theme/cluster name and summary.
Your cluster name and summary MUST be domain-agnostic, representing whatever field the papers belong to (e.g. medicine, law, commerce, physics, ML, etc.).

Paper Details in this Group:
{json.dumps(paper_details, indent=2, ensure_ascii=False)}

Instructions:
1. Output ONLY a valid JSON object matching the schema below.
2. Do not wrap the response in markdown code blocks.

JSON Output Schema:
{{
  "name": "A descriptive, professional theme name (max 5 words) summarizing this group",
  "description": "A 1-2 sentence description explaining the common research focus of this group"
}}

JSON Output:"""

        cluster_name = f"Cluster {idx + 1}"
        cluster_desc = "A grouped set of research documents."

        try:
            res = await model.generate_content_async(prompt)
            raw_text = res.text.strip()
            
            # Clean JSON markdown wrappers if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
            raw_text = raw_text.strip()
            
            data = json.loads(raw_text)
            cluster_name = data.get("name", cluster_name)
            cluster_desc = data.get("description", cluster_desc)
        except Exception as e:
            warnings.append(f"Could not generate AI label for cluster {idx + 1}: {e}")

        labeled_clusters.append({
            "cluster_id": idx + 1,
            "name": cluster_name,
            "description": cluster_desc,
            "papers": cluster_pids
        })

    return {
        "clusters": labeled_clusters,
        "warnings": warnings
    }

async def execute_timeline_generation(user_id: str = None) -> Dict:
    """
    Load all completed paper profiles, sort them chronologically by publication_year,
    and generate a high-level AI overview of the field's evolution.
    """
    metadata = load_metadata(user_id=user_id)
    all_paper_ids = list(metadata.get("papers", {}).keys())

    if not all_paper_ids:
        return {
            "timeline": [],
            "overview": "No papers uploaded to generate a timeline.",
            "warnings": ["No papers available in the library."]
        }

    timeline_items = []
    warnings = []

    # 1. Load profiles and extract year/objective
    for pid in all_paper_ids:
        profile = load_profile(pid, user_id=user_id)
        if not profile:
            warnings.append(f"Paper '{pid}' is missing its structured profile.")
            continue
            
        status = profile.get("profile_status", "failed")
        if status != "completed":
            warnings.append(f"Paper '{pid}' is not yet processed (status: {status}).")
            continue

        year = profile.get("publication_year")
        timeline_items.append({
            "paper_id": pid,
            "title": profile.get("title", pid),
            "publication_year": year,
            "authors": profile.get("authors", []),
            "objective": profile.get("objective", "Not stated")
        })

    if not timeline_items:
        return {
            "timeline": [],
            "overview": "No completed paper profiles are currently available.",
            "warnings": warnings
        }

    # 2. Sort chronologically: group papers with years first, sorted, then papers without years
    papers_with_year = [p for p in timeline_items if p["publication_year"] is not None]
    papers_without_year = [p for p in timeline_items if p["publication_year"] is None]

    # Sort ascending by year
    papers_with_year.sort(key=lambda x: x["publication_year"])
    sorted_timeline = papers_with_year + papers_without_year

    # 3. Call Gemini to synthesize a brief evolutionary overview
    overview_text = "Timeline generated successfully."
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL_NAME)
        prompt = f"""You are a senior academic research synthesist.
Analyze the following chronological timeline of research papers and write a 2-3 sentence overview summarizing the evolutionary trajectory of this research field.

Chronological Timeline:
{json.dumps(sorted_timeline, indent=2, ensure_ascii=False)}

Overview:"""
        res = await model.generate_content_async(prompt)
        overview_text = res.text.strip()
    except Exception as e:
        warnings.append(f"Could not generate AI timeline overview: {e}")

    return {
        "timeline": sorted_timeline,
        "overview": overview_text,
        "warnings": warnings
    }

async def execute_hierarchical_clustering(user_id: str = None) -> Dict:
    """
    Perform Agglomerative Hierarchical Clustering using average linkage,
    and dynamically request Gemini to label the resulting clusters.
    """
    metadata = load_metadata(user_id=user_id)
    all_paper_ids = list(metadata.get("papers", {}).keys())

    if not all_paper_ids:
        return {
            "clusters": [],
            "warnings": ["No papers uploaded to cluster."]
        }

    # Load global embeddings
    paper_embeddings = {}
    for pid in all_paper_ids:
        if user_id:
            emb_path = get_user_embeddings_dir(user_id) / f"{pid}.npy"
        else:
            emb_path = EMBEDDINGS_DIR / f"{pid}.npy"
        if emb_path.exists():
            try:
                emb = np.load(emb_path)
                if len(emb) > 0:
                    paper_embeddings[pid] = emb[0]
            except Exception as e:
                print(f"Error loading embedding for {pid}: {e}")

    pids = list(paper_embeddings.keys())
    n = len(pids)
    
    if n == 0:
        return {
            "clusters": [],
            "warnings": ["No valid paper embeddings found for clustering."]
        }

    # Group papers into hierarchy
    # We start with each paper as its own cluster
    current_clusters = [[pid] for pid in pids]

    def get_average_linkage_similarity(c1, c2):
        sims = []
        for p1 in c1:
            for p2 in c2:
                v1 = paper_embeddings[p1]
                v2 = paper_embeddings[p2]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    sims.append(np.dot(v1, v2) / (norm1 * norm2))
                else:
                    sims.append(0.0)
        return np.mean(sims) if sims else 0.0

    # Merge until a similarity threshold is hit (e.g. 0.65)
    min_similarity_threshold = 0.65
    
    while len(current_clusters) > 1:
        best_sim = -1.0
        merge_indices = (0, 0)
        
        for i in range(len(current_clusters)):
            for j in range(i + 1, len(current_clusters)):
                sim = get_average_linkage_similarity(current_clusters[i], current_clusters[j])
                if sim > best_sim:
                    best_sim = sim
                    merge_indices = (i, j)
                    
        if best_sim >= min_similarity_threshold:
            i, j = merge_indices
            current_clusters[i] = current_clusters[i] + current_clusters[j]
            current_clusters.pop(j)
        else:
            break

    # Ask Gemini to label each cluster
    labeled_clusters = []
    warnings = []
    model = genai.GenerativeModel(DEFAULT_MODEL_NAME)

    for idx, cluster_pids in enumerate(current_clusters):
        paper_details = []
        for pid in cluster_pids:
            profile = load_profile(pid, user_id=user_id)
            if profile:
                paper_details.append({
                    "paper_id": pid,
                    "title": profile.get("title", pid),
                    "research_topics": profile.get("research_topics", []),
                    "objective": profile.get("objective", "Not stated")
                })
            else:
                paper_details.append({
                    "paper_id": pid,
                    "title": pid,
                    "research_topics": [],
                    "objective": "Not stated"
                })

        prompt = f"""You are a senior academic research synthesist.
Analyze the following group of research papers formed via Hierarchical Clustering and generate a theme/cluster name and summary.
Your cluster name and summary MUST be domain-agnostic.

Paper Details in this Group:
{json.dumps(paper_details, indent=2, ensure_ascii=False)}

Instructions:
1. Output ONLY a valid JSON object matching the schema below.
2. Do not wrap the response in markdown code blocks.

JSON Output Schema:
{{
  "name": "A descriptive theme name (max 5 words) summarizing this hierarchical cluster",
  "description": "A 1-2 sentence description explaining the common research focus of this hierarchical group"
}}

JSON Output:"""

        cluster_name = f"Hierarchical Cluster {idx + 1}"
        cluster_desc = "A hierarchically grouped set of research documents."

        try:
            res = await model.generate_content_async(prompt)
            raw_text = res.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
            raw_text = raw_text.strip()
            data = json.loads(raw_text)
            cluster_name = data.get("name", cluster_name)
            cluster_desc = data.get("description", cluster_desc)
        except Exception as e:
            warnings.append(f"Could not generate AI label for hierarchical cluster {idx + 1}: {e}")

        labeled_clusters.append({
            "cluster_id": idx + 1,
            "name": cluster_name,
            "description": cluster_desc,
            "papers": cluster_pids
        })

    return {
        "clusters": labeled_clusters,
        "warnings": warnings
    }
