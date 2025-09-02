# build_index.py
import os
import fitz  # PyMuPDF
from rag import KnowledgeGraph
from llm import extract_triples_from_text, extract_descriptions_and_claims

PDF_PATH = "docs/constitution.pdf"
CHUNK_MINLEN = 50

os.makedirs("storage", exist_ok=True)
kg = KnowledgeGraph()

doc = fitz.open(PDF_PATH)
base_name = os.path.basename(PDF_PATH)

for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text").strip()
    if not text:
        continue
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= CHUNK_MINLEN]
    for idx, para in enumerate(paras):
        para_id = f"{base_name}_p{page_num}_{idx}"
        # extract triples using LLM
        triples = extract_triples_from_text(para)
        if not triples:
            # fallback: store paragraph itself as a node
            kg.add_fact(subject=para_id, relation="has_paragraph", obj=para_id, subj_text=para, obj_text=para, source=base_name)
            continue
        for t in triples:
            subj = t["subject"]
            rel = t["relation"]
            obj = t["object"]
            # get descriptive fields / claims (LLM)
            descr = extract_descriptions_and_claims(para, t)
            subj_desc = descr.get("subject_desc", "") or subject if (subject := subj) else subj
            obj_desc = descr.get("object_desc", "") or obj
            claims = descr.get("claims", [])
            # add to KG — prefer descriptive texts as node text
            kg.add_fact(subject=subj, relation=rel, obj=obj, subj_text=subj_desc, obj_text=obj_desc, claims=claims, source=f"{base_name}_p{page_num}")
# detect multi-level communities
kg.detect_multi_level(resolutions=[2.0, 1.0, 0.5])
# precompute community summaries (pick one or compute for all levels)
for level in [0, 1, 2]:
    try:
        kg.compute_and_store_community_summaries(level=level, top_texts_per_comm=25)
    except Exception as e:
        print(f"Warning: could not compute summaries for level {level}: {e}")

kg.persist()
print("✅ Index build complete.")
