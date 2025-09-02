# legal_bot_frontend.py
import os
import streamlit as st
from rag import KnowledgeGraph
from llm import generate_response
from speech_api import tts_long_text

st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("📄 AI Legal Assistant (Graph-RAG)")

# Initialize KG once
@st.cache_resource
def get_kg():
    return KnowledgeGraph()

kg = get_kg()

# session state for audio files
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

def delete_audio_files():
    for file_path in list(st.session_state.audio_files):
        try:
            os.remove(file_path)
        except Exception:
            pass
    st.session_state.audio_files = []

# Sidebar controls for query parameters
st.sidebar.header("Query settings")
level = st.sidebar.number_input("Community level", min_value=0, max_value=3, value=1, step=1)
top_communities = st.sidebar.number_input("Top communities to consider", min_value=1, max_value=10, value=3, step=1)
top_nodes_per_comm = st.sidebar.number_input("Top nodes per community", min_value=1, max_value=20, value=5, step=1)

st.header("🔍 Ask a Legal Question")
query = st.text_input("Enter your legal question:")

if st.button("Ask") and query.strip():
    # clean previous audio
    delete_audio_files()

    with st.spinner("Querying knowledge graph (community-first retrieval)..."):
        try:
            selection = kg.query(
                question=query,
                level=level,
                top_communities=top_communities,
                top_nodes_per_comm=top_nodes_per_comm
            )
        except Exception as e:
            st.error(f"KG query failed: {e}")
            st.stop()

    selected = selection.get("selected_communities", [])
    if not selected:
        st.info("No relevant communities found in the KG (make sure you've run build_index.py and computed community summaries).")
        st.stop()

    # Show retrieved communities + nodes
    st.subheader("📚 Retrieved community contexts")
    for comm in selected:
        cid = comm["community_id"]
        score = comm["score"]
        st.markdown(f"**Community {cid}** — score: {score:.3f}")
        with st.expander(f"Summary (community {cid})"):
            st.write(comm.get("community_summary", "—"))
        for node in comm["nodes"]:
            with st.expander(f"Node: {node['node']} — score {node['score']:.3f}"):
                st.write("Text:")
                st.write(node.get("text", ""))
                st.write("Supporting triples:")
                for t in node.get("triples", []):
                    st.write(f"- {t.get('subject')} —[{t.get('relation')}]-> {t.get('object')} (source: {t.get('source')})")

    # Prepare contexts and generate partial answers sequentially (avoids parallel LLM overload)
    community_contexts = []
    for comm in selected:
        ctx_chunks = []
        ctx_chunks.append(f"Community summary:\n{comm['community_summary']}")
        for node in comm["nodes"]:
            ctx_chunks.append(f"Node: {node['node']}\nText: {node['text']}\nTriples: {node['triples']}")
        community_contexts.append({"community_id": comm["community_id"], "context": ctx_chunks})

    partial_answers = []
    with st.spinner("Generating partial answers (one per community)..."):
        for c in community_contexts:
            try:
                part_ans = generate_response(query, c["context"])
            except Exception as e:
                part_ans = f"[Error generating partial answer for community {c['community_id']}: {e}]"
            partial_answers.append(part_ans)

    # Combine partial answers
    with st.spinner("Aggregating final answer from partial summaries..."):
        try:
            final_answer = generate_response(query, partial_answers)
        except Exception as e:
            st.error(f"Final aggregation failed: {e}")
            st.stop()

    # Convert to speech
    with st.spinner("Converting answer to audio..."):
        try:
            success, audio_or_err = tts_long_text(final_answer, base_filename="answer")
        except Exception as e:
            success = False
            audio_or_err = f"TTS error: {e}"

    # Display final answer
    st.subheader("🤖 AI Answer")
    st.write(final_answer)

    # Show partial answers (for transparency)
    if partial_answers:
        with st.expander("Partial answers (by community)"):
            for i, pa in enumerate(partial_answers, start=1):
                st.markdown(f"**Partial answer {i}:**")
                st.write(pa)

    # Handle audio playback & storage
    if success:
        st.session_state.audio_files = audio_or_err
        st.success("Audio generated.")
        for path in audio_or_err:
            try:
                with open(path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            except Exception as e:
                st.error(f"Could not play audio file {path}: {e}")
    else:
        st.error(f"Audio generation failed: {audio_or_err}")
