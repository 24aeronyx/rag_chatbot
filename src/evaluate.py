import json
import csv
import requests
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
TOP_K = 5
QUESTIONS_FILE = 'Data/generated-questions.json'
OUTPUT_CSV = 'Data/evaluated-qa.csv'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2:3b'
TIMEOUT_SEC = 10
MAX_QUESTIONS = 25  # Bisa diubah sesuai kebutuhan

# Inisialisasi
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_context(question, top_k=TOP_K, window=2):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    all_ids = results.get('ids', [[]])[0]
    all_docs = results.get('documents', [[]])[0]
    all_metas = results.get('metadatas', [[]])[0]

    if not all_docs:
        return []

    # Ambil window sebelum dan sesudah setiap dokumen utama
    unique_chunks = {}
    for i, (doc, meta) in enumerate(zip(all_docs, all_metas)):
        if not doc or len(doc.strip()) < 50:
            continue

        # Ambil indeks dari metadata (wajib ada di index)
        idx = meta.get('chunk_index')
        if idx is None:
            continue

        href = meta.get('href', 'unknown')
        name = meta.get('name', 'unknown')

        # Ambil semua chunk di dokumen yang sama
        related = collection.get(where={"href": href})
        all_related_docs = related['documents']
        all_related_metas = related['metadatas']

        for offset in range(-window, window + 1):
            j = idx + offset
            if 0 <= j < len(all_related_docs):
                text = all_related_docs[j]
                submeta = all_related_metas[j]
                if text and len(text.strip()) >= 50:
                    key = (submeta['href'], submeta['chunk_index'])
                    if key not in unique_chunks:
                        unique_chunks[key] = {
                            "name": submeta.get("name", name),
                            "href": submeta.get("href", href),
                            "text": text.strip()
                        }

    # Hanya ambil maksimal TOP_K dokumen terbaik berdasarkan urutan awal
    combined = list(unique_chunks.values())
    grouped = {}
    for chunk in combined:
        key = f"{chunk['name']}::{chunk['href']}"
        grouped.setdefault(key, []).append(chunk['text'])

    # Hitung skor "ranking awal" agar tetap hanya TOP_K
    top_keys = list(grouped.keys())[:top_k]
    final_docs = []
    for key in top_keys:
        name, href = key.split("::", 1)
        final_docs.append({
            "name": name,
            "href": href,
            "text": " ".join(grouped[key])
        })

    return final_docs

def ask_llama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }, timeout=TIMEOUT_SEC)

        if response.status_code == 200:
            resp_text = response.json().get("response", "").strip()
            if not resp_text:
                return "⚠️ Model tidak memberikan respons."
            return resp_text
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return "⚠️ Request ke Ollama API timeout."
    except Exception as e:
        return f"⚠️ Terjadi error saat memanggil Ollama API: {str(e)}"

def build_prompt(context_docs, question):
    context_lines = []
    for i, doc in enumerate(context_docs, 1):
        # Batasi panjang teks konteks untuk prompt agar tidak terlalu besar (misal 500 karakter)
        snippet = doc['text'][:500].replace('\n', ' ').strip()
        context_lines.append(f"[{i}] {doc['name']} - {doc['href']}\n{snippet}\n")

    context_block = "\n".join(context_lines) if context_lines else "Tidak ada konteks yang relevan ditemukan."

    return f"""
Kamu adalah asisten kesehatan profesional. Jawablah pertanyaan pengguna hanya berdasarkan informasi dalam konteks berikut.

=== Konteks Artikel Kesehatan ===
{context_block}
=== Akhir Konteks ===

Jika informasi tidak cukup, jawab: "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

Pertanyaan: {question}
Jawaban:
""".strip()

def evaluate_and_export_csv(questions, output_file):
    results = []
    reciprocal_ranks = []

    print("\n🔍 Evaluasi Manual MRR ke CSV\n")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'docs', 'rank', 'RR'])

        for i, item in enumerate(questions[:MAX_QUESTIONS]):
            question = item.get('question', '').strip()
            if not question:
                print(f"⚠️ Item ke-{i+1} tidak memiliki pertanyaan valid, dilewati.")
                continue

            sources = query_context(question)
            prompt = build_prompt(sources, question)
            answer = ask_llama(prompt)

            print(f"\n{i+1}. ❓ Pertanyaan: {question}")
            print(f"💬 Jawaban LLaMA:\n{answer}\n")
            if sources:
                for idx, source in enumerate(sources):
                    print(f"📄 [{idx+1}] {source['name']} - {source['href']}")
            else:
                print("📄 Tidak ada konteks relevan yang ditemukan.")

            while True:
                try:
                    rank_input = input(f"🏷️ Rank dokumen yang benar (1-{TOP_K}, atau 0 jika tidak relevan): ").strip()
                    rank = int(rank_input)
                    if 0 <= rank <= TOP_K:
                        break
                except ValueError:
                    pass
                print(f"⚠️ Masukkan angka antara 0 sampai {TOP_K}.")

            rr = 1.0 / rank if rank > 0 else 0.0
            reciprocal_ranks.append(rr)

            docs_str = "; ".join([f"{d['name']} ({d['href']})" for d in sources]) if sources else "Tidak ada dokumen"
            writer.writerow([question, answer, docs_str, rank, f"{rr:.4f}"])

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        writer.writerow([])
        writer.writerow(["", "", "", "MRR", f"{mrr:.4f}"])

    print(f"\n📁 CSV selesai disimpan ke: {output_file}")
    print(f"📊 Nilai MRR rata-rata: {mrr:.4f}")

if __name__ == "__main__":
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    evaluate_and_export_csv(questions, OUTPUT_CSV)
