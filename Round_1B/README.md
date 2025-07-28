# 🤖 Adobe Hackathon 2025 — Round 1B
## 🧬 Persona-Driven Document Intelligence

This project solves the 1B challenge using semantic embeddings to extract user-specific content from PDFs based on personas and goals.

---

### 🛠 Technologies
- Python
- FAISS for vector similarity search
- Sentence Transformers (Embeddings)
- Docker

---

### 📂 Folder Structure

Round_1B/
├── run.py
├── layout_parser.py
├── requirements.txt
├── input/
│ ├── challenge1b_input.json
│ └── pdfs/*.pdf
├── cache/
│ ├── embeddings.npz
│ └── faiss.index
├── Dockerfile

---

### 🚀 Run Instructions

Docker:
```bash
docker build -t adobe-round1b .
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output round1b

Local:bash
pip install -r requirements.txt
python run.py


🧩 Input Format
{
  "persona": "Health-Conscious",
  "goal": "Find low-fat breakfast options",
  "documents": ["Breakfast Ideas.pdf", "Dinner Ideas - Mains_1.pdf"]
}

📤 Output Format
{
  "persona": "Health-Conscious",
  "goal": "Find low-fat breakfast options",
  "matches": [
    {
      "document": "Breakfast Ideas.pdf",
      "text": "Oatmeal with fruits",
      "score": 0.89
    }
  ]
}
⚙️ How It Works
PDFs are parsed using layout_parser.py

Sentences are embedded using pretrained Sentence Transformers

FAISS indexes embeddings for fast similarity search

The best matching sentences are returned per persona-goal pair