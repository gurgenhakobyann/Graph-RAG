# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag import KnowledgeGraph
from llm import generate_response
import logging
from fastapi.responses import FileResponse
from speech_api import tts_long_text
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kg = KnowledgeGraph()

class QuestionRequest(BaseModel):
    question: str
    level: int = 1  # community level to use
    top_communities: int = 3
    top_nodes_per_comm: int = 5

@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # Map stage: get structured selected communities + top nodes
        selection = kg.query(question, level=request.level, top_communities=request.top_communities, top_nodes_per_comm=request.top_nodes_per_comm)
        selected = selection["selected_communities"]
        if not selected:
            return {"answer": "No relevant communities found in the KG."}

        # For each community create context (community_summary + top node texts/triples)
        community_contexts = []
        for comm in selected:
            ctx_chunks = []
            ctx_chunks.append(f"Community summary:\n{comm['community_summary']}")
            for node in comm["nodes"]:
                # include node text and supporting triples
                ctx_chunks.append(f"Node: {node['node']}\nText: {node['text']}\nTriples: {node['triples']}")
            community_contexts.append({"community_id": comm["community_id"], "context": ctx_chunks})

        # Use ThreadPool to generate partial answers in parallel
        partial_answers = []
        with ThreadPoolExecutor(max_workers=min(6, len(community_contexts))) as ex:
            futures = {ex.submit(generate_response, question, c["context"]): c for c in community_contexts}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    ans = fut.result()
                except Exception as e:
                    logger.exception("LLM generation for community failed")
                    ans = f"[Error generating summary for community {c['community_id']}]"
                partial_answers.append(ans)

        # Reduce stage: combine partial answers into final answer
        final_answer = generate_response(question, partial_answers)

        # TTS
        success, audio_or_err = tts_long_text(final_answer, base_filename="answer")
        if not success:
            raise Exception(audio_or_err)

        # sources & snippets for provenance
        sources = []
        snippets = []
        for comm in selected:
            for node in comm["nodes"]:
                for t in node["triples"]:
                    if t.get("source"):
                        sources.append(t["source"])
                snippets.append({"community": comm["community_id"], "node": node["node"], "text": node["text"]})

        return {
            "answer": final_answer,
            "audio_files": [str(p) for p in audio_or_err],
            "sources": list(set(sources)),
            "snippets": snippets
        }

    except Exception as e:
        logger.exception("Knowledge-graph query or generation failed.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail="Upload → knowledge-graph ingestion not yet implemented (use build_index.py).")

@app.get("/audio")
async def get_audio():
    return FileResponse("answer_0.wav", media_type="audio/wav", filename="answer_0.wav")
