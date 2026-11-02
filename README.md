# Retrieval Augmented Generation (RAG) — Webflyx

A hands-on implementation of a Retrieval Augmented Generation (RAG) pipeline built with Python, as part of the [Boot.dev Learn Retrieval Augmented Generation](https://www.boot.dev/courses/learn-retrieval-augmented-generation) course.

## Certificate of Completion

[![Boot.dev Learn Retrieval Augmented Generation certificate](https://qvault-webapp-dynamic-assets.storage.googleapis.com/certificates/738bd7e8-9457-4d18-9ea3-a80324dd23c1.jpeg?v=1793611734)](https://www.boot.dev/certificates/738bd7e8-9457-4d18-9ea3-a80324dd23c1)

## About the Project

This project explores how modern search engines work and how to build a complete RAG pipeline from scratch.

Using **Webflyx**, a fictional movie streaming service and movie dataset, the project progressively implements different retrieval techniques and combines them with Large Language Models (LLMs) to generate relevant, context-aware answers.

## What I Learned & Implemented

- **Preprocessing:** Cleaning, normalizing, and preparing datasets for search.
- **TF-IDF:** Building inverted indexes and implementing term weighting.
- **Keyword Search:** Implementing keyword-based retrieval and BM25 ranking.
- **Semantic Search:** Generating embeddings and retrieving documents using vector similarity.
- **Chunking:** Splitting documents into smaller, meaningful chunks.
- **Hybrid Search:** Combining keyword and semantic search using Reciprocal Rank Fusion (RRF).
- **LLMs:** Query expansion, rewriting, and correcting search intent.
- **Reranking:** Improving search relevance by reranking retrieved documents.
- **Evaluation:** Measuring retrieval quality and comparing search strategies.
- **Augmented Generation:** Using retrieved documents to generate summaries and citation-aware answers.
- **Agentic Systems:** Exploring tool calling and iterative LLM workflows.
- **Multimodal Search:** Using image embeddings and visual information to improve search.

## Tech Stack

- Python
- NumPy
- Sentence Transformers
- CLIP
- OpenAI SDK
- OpenRouter
- uv
- Argparse

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ogzer0day/Rag-Search-Engine.git
cd Rag-Search-Engine
```

Install dependencies:

```bash
uv sync
```

Configure your environment variables as required by the project.

## Key Takeaway

The goal wasn't just to use an existing RAG framework, but to understand what's happening under the hood: how documents are processed, indexed, retrieved, ranked, and ultimately used by LLMs to generate grounded responses.

**Built while learning, experimenting, and implementing the fundamentals of modern AI-powered search.**
