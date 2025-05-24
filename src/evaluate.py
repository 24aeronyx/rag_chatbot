import json
from tqdm import tqdm
import requests
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
TOP_K = 5
QUESTIONS_FILE = 'data/generated-questions.json'
OUTPUT_FILE = 'data/evaluated-qa.json'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2'

# Inisialisasi
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_context(question, top_k=TOP_K):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    grouped = {}
    for doc, meta in zip(docs, metadatas):
        href = meta.get("href", "unknown")
        name = meta.get("name", "unknown")
        key = f"{name}::{href}"
        grouped.setdefault(key, []).append(doc.strip())

    combined_docs = []
    for key, chunks in grouped.items():
        name, href = key.split("::", 1)
        combined_docs.append({
            "name": name,
            "href": href,
            "text": " ".join(chunks)
        })

    return combined_docs

def ask_llama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "").strip()
    return f"⚠️ Error {response.status_code}: {response.text}"

def build_prompt(context_docs, question):
    context_lines = []
    for i, doc in enumerate(context_docs, 1):
        context_lines.append(
            f"[{i}] {doc['name']} - {doc['href']}\n{doc['text']}\n"
        )

    context_block = "\n".join(context_lines)

    return f"""
Kamu adalah asisten kesehatan profesional. Jawablah pertanyaan pengguna hanya berdasarkan informasi dalam konteks berikut.

=== Konteks Artikel Kesehatan ===
{context_block}
=== Akhir Konteks ===

Jika informasi tidak cukup, jawab: "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

Pertanyaan: {question}
Jawaban:
""".strip()

def evaluate(questions):
    results = []
    reciprocal_ranks = []

    print("\n🔍 Evaluasi MRR Manual Berdasarkan Rank Dokumen\n")

    for i, item in enumerate(questions[:25]):
        question = item['question']
        sources = query_context(question)
        prompt = build_prompt(sources, question)
        answer = ask_llama(prompt)

        print(f"\n{i+1}. ❓ Pertanyaan: {question}")
        print(f"💬 Jawaban LLaMA:\n{answer}\n")
        for idx, source in enumerate(sources):
            print(f"📄 [{idx+1}] {source['name']} - {source['href']}")

        while True:
            try:
                rank = int(input("🏷️  Rank dokumen yang benar (1-5, atau 0 jika tidak relevan): ").strip())
                if 0 <= rank <= TOP_K:
                    break
            except ValueError:
                pass
            print("⚠️ Masukkan angka antara 0 sampai 5.")

        results.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "rank": rank
        })

        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return results, mrr

if __name__ == "__main__":
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results, mrr_score = evaluate(questions)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Evaluasi selesai. MRR: {mrr_score:.4f}")
    print(f"📁 Hasil disimpan ke: {OUTPUT_FILE}")
