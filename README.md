# ResearchLens

AI-powered research paper analysis and research gap identification system.

---

# Overview

ResearchLens is an AI-powered platform that analyzes hundreds of research papers and helps users understand an entire research domain rather than a single paper.

Unlike traditional **"Chat with PDF"** applications, ResearchLens focuses on:

* Discovering research trends
* Identifying research gaps
* Comparing methodologies
* Evaluating paper quality
* Understanding domain evolution

The goal is to help students, researchers, and engineers quickly understand a field without manually reading hundreds of papers.

---

# Problem Statement

Most AI paper tools today work like this:

1. Upload one paper
2. Ask questions
3. Get answers

This approach does not help users understand an entire research area.

For example, users researching topics like:

* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* AI Agents
* Computer Vision

may need to manually read hundreds of papers.

ResearchLens automatically processes a large collection of papers and generates insights about the field as a whole.

---

# Key Features

## 1. Research Timeline Generator

Automatically identifies important milestones and organizes papers chronologically.

### Example Output

```text
2020 → GPT-3
2021 → Instruction Tuning
2022 → Chain of Thought
2023 → Retrieval-Augmented Generation (RAG)
2024 → Multi-Agent Systems
2025 → Deep Research Agents
```

### Benefits

* Understand evolution of a field
* Learn important breakthroughs
* Identify emerging trends

---

## 2. Research Cluster Analysis

Groups papers into related categories using embeddings and clustering algorithms.

### Example Clusters

* Prompt Engineering
* RAG Systems
* AI Agents
* Evaluation Frameworks
* Long Context Models

### Visualization

Interactive graphs showing:

* Papers
* Relationships
* Cluster memberships

---

## 3. AI Paper Critic

Instead of only summarizing papers, the system evaluates them critically.

### Example Output

```text
Paper: Advanced Agentic Retrieval Systems

Novelty Score: 8/10
Evidence Strength: 7/10
Reproducibility: Medium

Strengths:
- Strong benchmark results
- Clear methodology

Weaknesses:
- Limited datasets
- Missing ablation studies
- Small sample size
```

### Benefits

Provides deeper insights than simple summaries.

---

## 4. Paper Comparison Engine

Users can compare two papers side-by-side.

### Example

#### Paper A

* Uses Vector Retrieval

#### Paper B

* Uses Graph Retrieval

### Comparison

#### Advantages of Paper A

* Faster retrieval
* Simpler implementation

#### Advantages of Paper B

* Better relationship understanding
* More context-aware retrieval

#### Tradeoffs

* Complexity
* Accuracy
* Cost
* Scalability

---

## 5. Research Gap Finder

One of the most valuable features of the platform.

After analyzing all papers, the system identifies underexplored research opportunities.

### Example Output

```text
Potential Research Opportunities

1. Graph RAG for Healthcare
2. Agent Evaluation Benchmarks
3. Long Context Retrieval Systems
4. Multi-Agent Scientific Research
5. Retrieval-Augmented Reasoning Systems
```

This feature helps researchers discover potential project ideas and unexplored areas.

---

# System Architecture

```text
PDF Upload
    ↓
PDF Processing
    ↓
Metadata Extraction
    ↓
Text Chunking
    ↓
Embeddings Generation
    ↓
Vector Database
    ↓
Analysis Engine
    ↓
Large Language Model
    ↓
Reports & Visualizations
```

---

# Technology Stack

## Backend

* Python
* FastAPI

### Why FastAPI?

* Easy AI integration
* Large ecosystem
* Fast development

---

## Database

### PostgreSQL

Stores:

* Papers
* Metadata
* Users
* Analysis results

---

## Vector Database

Possible options:

* ChromaDB
* FAISS
* Weaviate

Used for storing paper embeddings and semantic search.

---

## PDF Processing

Libraries:

* PyMuPDF
* pdfplumber

Responsibilities:

* Extract text
* Extract metadata
* Extract references

---

## AI Models

### OpenAI

* GPT-4o
* GPT-5

### Anthropic

* Claude Sonnet

### Open Source Models

* Qwen
* Llama

Used for:

* Summarization
* Critique generation
* Trend analysis
* Research gap discovery

---

## Visualization

### Libraries

* NetworkX
* Plotly

Used for:

* Research clusters
* Citation relationships
* Topic evolution

---

# Core Engineering Challenges

## Challenge 1: Large Papers

Research papers can exceed 50 pages.

### Problem

Entire papers cannot fit into model context windows.

### Solution

Chunking strategy:

* Abstract
* Introduction
* Methodology
* Results
* Conclusion

Then perform hierarchical summarization.

---

## Challenge 2: Hundreds of Papers

### Problem

100–500 papers cannot fit into a single context window.

### Solution

Multi-level summarization:

```text
Paper Summary
      ↓
Cluster Summary
      ↓
Domain Summary
```

This allows the model to understand the research field at multiple abstraction levels.

---

## Challenge 3: Structured Understanding

### Problem

Comparing papers directly from raw text is difficult.

### Solution

Extract structured information:

```json
{
  "title": "...",
  "problem": "...",
  "method": "...",
  "dataset": "...",
  "results": "...",
  "limitations": "...",
  "future_work": "..."
}
```

Structured representations make comparisons significantly easier.

---

# Stretch Goal: Research Debate Mode

A highly impressive feature for interviews and demonstrations.

For every paper:

## Agent A — Defender

Questions:

* Why is this work important?
* What are its strengths?
* What contributions does it make?

## Agent B — Critic

Questions:

* What assumptions are weak?
* What experiments are missing?
* What limitations exist?

## Final Judge

Produces:

* Strengths
* Weaknesses
* Confidence level
* Final verdict

This demonstrates multi-agent reasoning and advanced AI workflows.

---

# Development Roadmap

## Phase 1 (2–3 Weeks)

### Build

* PDF upload
* PDF parsing
* Embeddings
* Semantic search

### Goal

Search across hundreds of papers.

---

## Phase 2 (2 Weeks)

### Build

* Paper summaries
* Timeline generation
* Cluster analysis

### Goal

Understand research trends.

---

## Phase 3 (2 Weeks)

### Build

* Paper critic
* Paper comparison

### Goal

Generate deeper insights.

---

## Phase 4 (1–2 Weeks)

### Build

* Research gap finder
* Visualizations

### Goal

Produce actionable insights.

---

## Phase 5 (Optional)

### Build

* Debate mode
* Multi-agent analysis

### Goal

Create a standout project for interviews and demonstrations.

---

# Skills Demonstrated

This project showcases:

* Python Development
* FastAPI
* LLM Integration
* RAG Systems
* Embeddings
* Vector Databases
* Information Retrieval
* Data Engineering
* Graph Analytics
* Agentic AI Systems
* System Design

These are highly relevant skills for AI, Backend, and Software Engineering roles.

---

# Phase 3 Update: Secure User Authentication & Data Isolation

ResearchLens now supports secure user authentication and absolute user-level data isolation.

## Security Architecture

1. **Authentication Flow**:
   - Users register via `POST /auth/register` (using hashed passwords via `bcrypt`).
   - Users login via `POST /auth/login` to obtain a JWT token.
   - All protected endpoints extract user identity via `get_current_user()` dependency.

2. **Data Isolation**:
   - **PostgreSQL Database**: Persistent metadata for users and uploaded papers.
   - **ChromaDB Filtering**: All search queries are scoped using `user_id` metadata filters: `where={"user_id": current_user.id}`.
   - **Path Validation**: Files are uploaded and stored within path-validated directories under `data/users/<user_id>/`.

3. **Legacy Migration**:
   - On server startup, legacy unowned paper files are automatically migrated to a disabled system user `legacy@researchlens.local` and metadata is updated.

## Running Tests

To verify user data isolation, run the test suite:
```bash
.\venv\Scripts\python.exe -m pytest tests/test_auth_isolation.py -s
```
