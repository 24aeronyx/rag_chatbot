import json
import requests
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
TOP_K = 5
QUESTIONS_FILE = 'Data/generated-questions.json'
OUTPUT_JSON = 'Data/ragas_dataset.json'
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

    all_docs = results.get('documents', [[]])[0]
    all_metas = results.get('metadatas', [[]])[0]

    if not all_docs:
        return []

    unique_chunks = {}
    for i, meta in enumerate(all_metas):
        idx = meta.get('chunk_index')
        href = meta.get('href', 'unknown')
        name = meta.get('name', 'unknown')
        if idx is None:
            continue

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

    combined = list(unique_chunks.values())
    # Hanya ambil maksimal top_k
    return combined[:top_k]

def build_prompt(context_docs, question):
    context_lines = []
    for i, doc in enumerate(context_docs, 1):
        snippet = doc['text'][:300].replace('\n', ' ').strip()
        context_lines.append(f"[{i}] {doc['name']} - {doc['href']}\n{snippet}\n")

    context_block = "\n".join(context_lines) if context_lines else "Tidak ada konteks yang relevan ditemukan."

    prompt = f"""
Kamu adalah asisten kesehatan profesional. Berdasarkan informasi berikut, bantu jawab pertanyaan pengguna secara langsung dan profesional. Kamu juga bisa mempertimbangkan konteks yang ada di sekitarnya sebelum menjawab.

=== Informasi milikmu ===
{context_block}
=== Akhir Informasi ===

Pertanyaan pengguna: "{question}"
Jawaban berdasarkan Informasi yang kamu miliki:
Jika tidak ada informasi yang mendukung, jawab dengan sopan: "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

Jawaban:
""".strip()

    return prompt

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

def build_evaluation_dataset(questions, max_questions=MAX_QUESTIONS):
    dataset = []
    for i, item in enumerate(questions[:max_questions]):
        question = item.get('question', '').strip()
        if not question:
            print(f"⚠️ Item ke-{i+1} tidak memiliki pertanyaan valid, dilewati.")
            continue

        contexts = query_context(question)
        prompt = build_prompt(contexts, question)
        answer = ask_llama(prompt)

        dataset.append({
            "question": question,
            "contexts": contexts,
            "answer": answer
        })

        print(f"[{i+1}/{max_questions}] Pertanyaan diproses.")

    return dataset

if __name__ == "__main__":
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    dataset = build_evaluation_dataset(questions, MAX_QUESTIONS)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset evaluasi selesai disimpan di: {OUTPUT_JSON}")
