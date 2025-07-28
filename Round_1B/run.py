import os
import json
import fitz
import numpy as np
import faiss
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).isoformat()
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from layout_parser import detect_headings, load_fallback


INPUT_JSON = "input/challenge1b_input.json"
PDF_DIR = "input/pdfs"
OUTPUT_JSON = "output/challenge1b_output.json"
EMBED_MODEL = "intfloat/e5-small-v2"  
TOP_K = 5

with open(INPUT_JSON, encoding="utf-8") as f:
    cfg = json.load(f)


persona = cfg.get("persona", "")
job = cfg.get("job_to_be_done", "")

if isinstance(persona, dict):
    persona_text = persona.get("text") or persona.get("description") or json.dumps(persona)
else:
    persona_text = str(persona)

if isinstance(job, dict):
    job_text = job.get("task") or job.get("description") or json.dumps(job)
else:
    job_text = str(job)

query = f"{persona_text.strip()} {job_text.strip()}".strip()


try:
    embedder = SentenceTransformer(EMBED_MODEL)
except Exception:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

qry_emb = normalize(embedder.encode([query]), axis=1).astype("float32")


fallback = load_fallback()
pdf_paths = [os.path.join(PDF_DIR, d["filename"]) for d in cfg["documents"]]
all_sections = []

def extract_full_sections(pdf_path, fallback):
    doc = fitz.open(pdf_path)
    texts_by_page = [page.get_text("text") for page in doc]
    headings = detect_headings(pdf_path, fallback=fallback)
    headings.sort(key=lambda h: (h["page"], h["y0"]))
    sections = []

    for i, h in enumerate(headings):
        start_page = h["page"] - 1
        end_page = len(doc)
        if i + 1 < len(headings):
            end_page = headings[i + 1]["page"] - 1

        content = []
        for p in range(start_page, end_page):
            content.append(texts_by_page[p])

        section_text = "\n".join(content).strip()
        if len(section_text) < 30:
            continue

        sections.append({
            "document": os.path.basename(pdf_path),
            "section_title": h["text"],
            "page_number": h["page"],
            "content": section_text
        })
    return sections

for pdf_path in pdf_paths:
    all_sections.extend(extract_full_sections(pdf_path, fallback))

section_texts = [f"{s['section_title']} {s['content']}" for s in all_sections]
sec_embeds = normalize(embedder.encode(section_texts), axis=1).astype("float32")


query_keywords = [w for w in query.lower().split() if len(w) > 4]
scored_sections = []

for i, s in enumerate(all_sections):
    emb = sec_embeds[i]
    sim = float(np.dot(emb, qry_emb.T).item())
    keyword_boost = sum(1 for k in query_keywords if k in s["content"].lower()) * 0.05
    score = sim + keyword_boost
    scored_sections.append({**s, "score": score})

from collections import defaultdict
per_doc = defaultdict(list)
for s in sorted(scored_sections, key=lambda x: -x["score"]):
    if len(per_doc[s["document"]]) < 1:
        per_doc[s["document"]].append(s)

diverse_sections = [s[0] for s in per_doc.values()]
final_sections = sorted(diverse_sections, key=lambda x: -x["score"])[:TOP_K]


metadata = {
    "input_documents": [d["filename"] for d in cfg["documents"]],
    "persona": cfg["persona"],
    "job_to_be_done": cfg["job_to_be_done"],
    "processing_timestamp": datetime.utcnow().isoformat()
}

out = {
    "metadata": metadata,
    "extracted_sections": [
        {
            "document": s["document"],
            "section_title": s["section_title"],
            "importance_rank": i + 1,
            "page_number": s["page_number"]
        } for i, s in enumerate(final_sections)
    ],
    "subsection_analysis": [
        {
            "document": s["document"],
            "refined_text": s["content"],
            "page_number": s["page_number"]
        } for s in final_sections
    ]
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("✅ Output saved to", OUTPUT_JSON)
