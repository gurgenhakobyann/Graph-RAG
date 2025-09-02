# llm.py
import os
import logging
import json
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)

LEGAL_PROMPT_TEMPLATE = """
You are a highly knowledgeable and professional legal assistant specializing exclusively in constitutional law. 
Use only the given context to answer clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

def generate_response(question: str, context_chunks: List[str], model: str = "llama-3.1-8b-instant", temperature: float = 0.1, max_tokens: int = 512) -> str:
    """
    Generic wrapper that composes context and calls LLM.
    context_chunks: list of strings (already formatted)
    """
    context = "\n\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(context_chunks)) or "No context available."
    prompt = LEGAL_PROMPT_TEMPLATE.format(context=context, question=question.strip())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a constitutional law assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Groq SDK call failed")
        raise RuntimeError(f"LLM generation error: {e}")

# --- Extract triples: LLM MUST return valid JSON list of dicts ---
def extract_triples_from_text(text: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> List[Dict]:
    prompt = f"""
You are a legal text analysis expert. Extract subject-relation-object triples from the following legal paragraph.
Return a VALID JSON array of objects in the form:
[{{"subject": "...", "relation": "...", "object": "..." }}, ...]
Do not output anything else.

Text:
\"\"\"{text.strip()}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        output = response.choices[0].message.content.strip()
        # attempt to extract json substring
        try:
            data = json.loads(output)
            if isinstance(data, list):
                valid = [d for d in data if isinstance(d, dict) and {"subject", "relation", "object"} <= set(d.keys())]
                return valid
        except Exception:
            # try to find first "[" ... "]" substring
            import re
            m = re.search(r"(\[.*\])", output, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    valid = [d for d in data if isinstance(d, dict) and {"subject", "relation", "object"} <= set(d.keys())]
                    return valid
                except Exception:
                    logger.warning("Could not parse JSON from LLM triples output.")
        return []
    except Exception as e:
        logger.warning(f"Triple extraction failed: {e}")
        return []

# --- Extract descriptions/claims for a triple (used at index-time) ---
def extract_descriptions_and_claims(paragraph: str, triple: Dict, model: str = "llama-3.3-70b-versatile", temperature: float = 0.2) -> Dict:
    """
    Returns dict like:
      {
        "subject_desc": "short text",
        "object_desc": "short text",
        "claims": ["claim1", "claim2"]
      }
    Output should be valid JSON.
    """
    prompt = f"""
You are a legal NLP assistant. For the following paragraph and extracted triple, produce:
1) short description of the SUBJECT (one sentence)
2) short description of the OBJECT (one sentence)
3) a small list of key claims or assertions present in the paragraph (max 5)

Return a VALID JSON object:
{{ "subject_desc": "...", "object_desc": "...", "claims": ["...", ...] }}

Paragraph:
\"\"\"{paragraph.strip()}\"\"\"

Triple:
{json.dumps(triple)}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512
        )
        output = response.choices[0].message.content.strip()
        try:
            data = json.loads(output)
            return {
                "subject_desc": data.get("subject_desc", ""),
                "object_desc": data.get("object_desc", ""),
                "claims": data.get("claims", []) or []
            }
        except Exception:
            # try to extract JSON substring
            import re
            m = re.search(r"(\{.*\})", output, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                    return {
                        "subject_desc": data.get("subject_desc", ""),
                        "object_desc": data.get("object_desc", ""),
                        "claims": data.get("claims", []) or []
                    }
                except Exception:
                    logger.warning("Failed to parse JSON from LLM descriptions output.")
        return {"subject_desc": "", "object_desc": "", "claims": []}
    except Exception as e:
        logger.warning(f"extract_descriptions_and_claims failed: {e}")
        return {"subject_desc": "", "object_desc": "", "claims": []}

# --- Summarize list of texts (returns string summary) ---
def summarize_texts(texts: List[str], instruction: str = "Summarize the texts focusing on main claims and entities.", model: str = "llama-3.3-70b-versatile", temperature: float = 0.1) -> str:
    # join carefully to avoid overlong prompt
    numbered = "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"{instruction}\n\nTexts:\n{numbered}\n\nProvide a concise summary (one paragraph)."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"summarize_texts failed: {e}")
        return "Summary unavailable."
