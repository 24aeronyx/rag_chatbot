import json
import pandas as pd
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain_core.documents import Document

# Konfigurasi
PERSIST_DIR = './embeddings'
QUESTIONS_FILE = 'Data/ragas_dataset.json'
OUTPUT_CSV = 'Data/evaluated-qa.csv'
MODEL_NAME = 'llama3.2:3b'

# === Load komponen ===

# Embedding manual dari SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Vector store dari Chroma (gunakan wrapper kosong karena kita embed sendiri)
vectordb = Chroma(persist_directory=PERSIST_DIR)

# Load LLM lokal dari Ollama
llm = Ollama(model=MODEL_NAME)

# Load dataset dari JSON
with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Format ke struktur untuk ragas
formatted = []
for item in data:
    if not all(key in item for key in ("question", "answer", "contexts", "ground_truths")):
        continue

    # pastikan contexts adalah list string
    context_texts = [ctx["text"] if isinstance(ctx, dict) else ctx for ctx in item["contexts"]]

    formatted.append({
        "question": item["question"],
        "answer": item["answer"],
        "contexts": context_texts,
        "ground_truths": item["ground_truths"]
    })

# Konversi ke HuggingFace Dataset
ragas_dataset = Dataset.from_list(formatted)

# Jalankan evaluasi dengan LLM lokal
results = evaluate(
    ragas_dataset,
    metrics=[context_precision, answer_relevancy, faithfulness],
    llm=llm
)

# Simpan ke CSV
df = results.to_pandas()
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Evaluasi selesai. Hasil disimpan ke: {OUTPUT_CSV}")
