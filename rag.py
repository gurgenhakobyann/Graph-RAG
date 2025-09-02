# rag.py
import os
import pickle
import logging
from typing import List, Dict, Optional
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import igraph as ig
import leidenalg
from llm import extract_triples_from_text, extract_descriptions_and_claims, summarize_texts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GRAPH_PATH = "storage/kg_graph.pkl"

class KnowledgeGraph:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.graph = nx.DiGraph()
        self.model = SentenceTransformer(embed_model_name)
        # graph-level storage for precomputed community summaries per level
        # stored inside graph.graph to persist with pickle
        if os.path.exists(GRAPH_PATH):
            try:
                with open(GRAPH_PATH, "rb") as f:
                    self.graph = pickle.load(f)
                logger.info(f"Loaded existing KG from {GRAPH_PATH}")
            except Exception as e:
                logger.warning(f"Could not load KG pickle: {e}. Starting fresh.")
        else:
            logger.info("No existing KG found; starting with an empty graph.")

    # --- add_fact: now supports node text, descriptions and claims ---
    def add_fact(self,
                 subject: str,
                 relation: str,
                 obj: str,
                 subj_text: Optional[str] = None,
                 obj_text: Optional[str] = None,
                 claims: Optional[List[str]] = None,
                 source: Optional[str] = None):
        """
        Adds an edge (subject -> object). Stores textual content and embeddings for nodes.
        subj_text / obj_text: textual content to embed (prefer real paragraph text).
        """
        # ensure nodes exist
        if subject not in self.graph.nodes:
            self.graph.add_node(subject)
        if obj not in self.graph.nodes:
            self.graph.add_node(obj)

        # add edge with metadata
        self.graph.add_edge(subject, obj, relation=relation, source=source, claims=claims or [])

        # store textual attributes
        if subj_text:
            self.graph.nodes[subject]["text"] = subj_text
        if obj_text:
            self.graph.nodes[obj]["text"] = obj_text

        # ensure we have an embedding for nodes (embed stored 'text' if available, otherwise node id)
        for node in [subject, obj]:
            if "embedding" not in self.graph.nodes[node]:
                text_to_embed = self.graph.nodes[node].get("text", node)
                emb = self.model.encode(text_to_embed, convert_to_numpy=True)
                # store as a numpy array (will be pickled)
                self.graph.nodes[node]["embedding"] = emb
        logger.info(f"Added triple: ({subject}) -[{relation}]-> ({obj})")

    # --- Query inside a given set of nodes (helper) ---
    def _rank_nodes_by_query(self, node_list: List[str], q_emb: np.ndarray, top_k: int):
        if not node_list:
            return []
        node_embs = np.array([self.graph.nodes[n]["embedding"] for n in node_list])
        # normalize (guard against zero-norm)
        nb_norms = np.linalg.norm(node_embs, axis=1, keepdims=True)
        nb_norms[nb_norms == 0] = 1e-10
        node_embs = node_embs / nb_norms

        qn = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        scores = node_embs @ qn
        top_idxs = np.argsort(scores)[-top_k:][::-1]
        results = [(node_list[idx], float(scores[idx])) for idx in top_idxs]
        return results

    # --- Multi-level Leiden partitioning ---
    def detect_multi_level(self, resolutions: List[float] = [2.0, 1.0, 0.5]):
        """
        Runs Leiden partitioning with multiple resolutions. Stores community ids on nodes:
            node['community_l{level}'] = integer community id
        Also stores graph.graph['community_partitions'] = { level: {community_id: [node_names] } }
        """
        logger.info("Running multi-level Leiden community detection...")
        undirected = self.graph.to_undirected()
        ig_graph = ig.Graph.TupleList(undirected.edges(), directed=False)

        partitions = {}
        for level, res in enumerate(resolutions):
            partition = leidenalg.find_partition(ig_graph,
                                                 leidenalg.RBConfigurationVertexPartition,
                                                 resolution_parameter=res)
            partitions[level] = {}
            logger.info(f"Level {level} (res={res}) → {len(partition)} communities")
            for community_id, community in enumerate(partition):
                node_names = [ig_graph.vs[v]["name"] for v in community]
                partitions[level][community_id] = node_names
                # assign to nodes
                for node in node_names:
                    self.graph.nodes[node][f"community_l{level}"] = community_id

        # persist partitions mapping in graph metadata
        self.graph.graph["community_partitions"] = partitions
        # initialize storage for precomputed summaries per level
        self.graph.graph.setdefault("community_summaries", {})
        logger.info("Multi-level Leiden finished.")

    # --- Precompute community summaries (index-time) ---
    def compute_and_store_community_summaries(self, level: int = 1, top_texts_per_comm: int = 20):
        """
        For each community at specified 'level', collect texts from nodes, produce a summary via LLM,
        compute embedding of that summary and store:
            self.graph.graph['community_summaries'][level][community_id] = {
                'summary': str, 'embedding': np.array, 'nodes': [...]
            }
        """
        partitions = self.graph.graph.get("community_partitions", {})
        if level not in partitions:
            raise ValueError(f"No partition at level {level}; run detect_multi_level() first.")

        logger.info(f"Computing community summaries for level {level} ...")
        comms = partitions[level]
        comm_summaries = {}
        for cid, nodes in comms.items():
            # gather node texts (prefer node['text'])
            texts = []
            for n in nodes:
                t = self.graph.nodes[n].get("text")
                if t:
                    texts.append(t)
            if not texts:
                # nothing to summarize
                continue
            # limit length to reasonable amount (top_texts_per_comm)
            texts_sample = texts[:top_texts_per_comm]
            # call LLM summarizer (questionless summarization)
            summary = summarize_texts(texts_sample, instruction="Summarize the following texts succinctly, focusing on main claims and entities.")
            # embed community summary
            comm_emb = self.model.encode(summary, convert_to_numpy=True)
            comm_summaries[cid] = {"summary": summary, "embedding": comm_emb, "nodes": nodes}
        # store into graph metadata
        self.graph.graph.setdefault("community_summaries", {})
        self.graph.graph["community_summaries"][level] = comm_summaries
        logger.info(f"Stored {len(comm_summaries)} community summaries at level {level}.")

    # --- Query flow: community-first map-reduce ---
    def query(self,
              question: str,
              level: int = 1,
              top_communities: int = 3,
              top_nodes_per_comm: int = 5) -> Dict:
        """
        Map-Reduce style query:
          1) embed query
          2) score community summaries at chosen level, pick top communities
          3) inside each selected community pick top nodes by node-embedding
          4) return structured info to be summarized by LLM
        Returns dict with:
          {
            'question': ...,
            'selected_communities': [ (cid, score, summary_text, top_nodes_info), ... ]
          }
        top_nodes_info: [ {node, score, text, edges}, ... ]
        """
        # step 0: quick guards
        comm_summaries = self.graph.graph.get("community_summaries", {}).get(level)
        if not comm_summaries:
            raise RuntimeError(f"No precomputed community summaries for level {level}. Run compute_and_store_community_summaries(level).")

        q_emb = self.model.encode(question, convert_to_numpy=True)
        # build arrays for scoring
        comm_ids = list(comm_summaries.keys())
        comm_embs = np.array([comm_summaries[c]["embedding"] for c in comm_ids])
        comm_embs_norms = np.linalg.norm(comm_embs, axis=1, keepdims=True)
        comm_embs_norms[comm_embs_norms == 0] = 1e-10
        comm_embs = comm_embs / comm_embs_norms
        qn = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        scores = comm_embs @ qn
        top_comm_idxs = np.argsort(scores)[-top_communities:][::-1]

        selected = []
        for idx in top_comm_idxs:
            cid = comm_ids[idx]
            score = float(scores[idx])
            comm_entry = comm_summaries[cid]
            # choose top nodes within this community
            node_candidates = comm_entry["nodes"]
            top_nodes = self._rank_nodes_by_query(node_candidates, q_emb, top_nodes_per_comm)
            # prepare node info
            node_infos = []
            for node_name, node_score in top_nodes:
                text = self.graph.nodes[node_name].get("text", "")
                # collect outgoing edges as supporting triples
                outs = []
                for nbr in self.graph.successors(node_name):
                    ed = self.graph.get_edge_data(node_name, nbr)
                    outs.append({
                        "subject": node_name,
                        "relation": ed.get("relation"),
                        "object": nbr,
                        "source": ed.get("source"),
                        "claims": ed.get("claims", [])
                    })
                node_infos.append({"node": node_name, "score": node_score, "text": text, "triples": outs})
            selected.append({
                "community_id": cid,
                "score": score,
                "community_summary": comm_entry["summary"],
                "nodes": node_infos
            })

        return {"question": question, "selected_communities": selected}

    # --- Persist graph (pickle) ---
    def persist(self):
        os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info(f"KG persisted to {GRAPH_PATH}")
