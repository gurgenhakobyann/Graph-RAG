# AI-Lawyer
📚 GraphRAG-based Legal Assistant

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) pipeline inspired by the GraphRAG paper (arXiv:2404.16130)
. It combines knowledge graph construction, community detection, and large language models (LLMs) to provide structured, explainable answers to complex legal queries.

🚀 Features

Document Ingestion & Chunking → splits legal texts into manageable units.

Entity & Relation Extraction → identifies key concepts (nodes) and relations (edges) using LLMs.

Claims Extraction → attaches natural-language claims as supporting evidence for each edge.

Knowledge Graph Construction → builds a graph of entities and relationships, merging duplicates.

Leiden Community Detection → clusters the graph into semantically coherent communities.

Community Summarization → generates higher-level summaries for each cluster, enabling hierarchical reasoning.

Semantic Embeddings → creates embeddings for node text, claims, and community summaries for retrieval.

Query Answering → combines node embeddings, graph structure, and community summaries to retrieve relevant evidence and generate accurate, explainable answers.

🧠 How it Works

Ingest documents → split into chunks.

Extract triples (subject, relation, object, claims) with LLM.

Build knowledge graph → nodes = entities, edges = relations + claims.

Merge & compress duplicates to maintain a clean graph.

Cluster graph with the Leiden algorithm → detect communities.

Summarize communities hierarchically (multi-level).

Embed nodes & communities → enable semantic search.

Query phase → user query is embedded → relevant nodes, edges, and community summaries are retrieved → LLM generates the final grounded answer.

🔍 Example

Query: "Does Armenia guarantee local self-governance?"

Retrieved node: (Local self-governance)

Edge: (Local self-governance) -[shall be guaranteed in]-> (Republic of Armenia)

Claim: "The Constitution states that local self-governance shall be guaranteed in the Republic of Armenia."

Community Summary: "This cluster describes constitutional guarantees of governance in Armenia."

Answer (LLM):
Yes, the Constitution of the Republic of Armenia guarantees local self-governance.

📦 Tech Stack

Python

NetworkX (graph construction & clustering)

Leidenalg (community detection)

SentenceTransformers (embeddings)

Groq / LLM API (entity, relation, and claim extraction + summarization)
