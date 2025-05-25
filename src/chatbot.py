import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import sys
import time

# Setup
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

# Init ChromaDB & model embedding
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_with_context_expansion(question, top_k=5, window=2, max_prompt_chunks=5):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])

    all_collection = collection.get(include=["documents", "metadatas"])
    full_ids = all_collection['ids']
    full_docs = all_collection['documents']
    full_metas = all_collection['metadatas']
    id_to_index = {id_: idx for idx, id_ in enumerate(full_ids)}

    base_ids = []
    for doc in results['documents'][0]:
        try:
            idx = full_docs.index(doc)
            base_ids.append(full_ids[idx])
        except ValueError:
            continue

    # Kumpulkan semua kandidat dari hasil ekspansi window
    candidate_chunks = []
    used = set()

    for base_id in base_ids:
        idx = id_to_index.get(base_id)
        if idx is None:
            continue

        for offset in range(-window, window + 1):
            i = idx + offset
            if i < 0 or i >= len(full_docs):
                continue
            chunk_id = full_ids[i]
            if chunk_id in used:
                continue
            used.add(chunk_id)
            chunk_text = full_docs[i]
            meta = full_metas[i]
            sim_score = float(embedder.encode([question, chunk_text])[0].dot(embedder.encode(chunk_text)))
            candidate_chunks.append((sim_score, chunk_text, meta))

    # Urutkan berdasarkan similarity score dan ambil maksimal `max_prompt_chunks`
    candidate_chunks.sort(reverse=True, key=lambda x: x[0])
    top_chunks = candidate_chunks[:max_prompt_chunks]

    docs = [doc for _, doc, _ in top_chunks]
    metas = [meta for _, _, meta in top_chunks]
    return docs, metas



def clean_semicolon_text(text):
    if ";" in text:
        items = [i.strip() for i in text.split(";") if i.strip()]
        if len(items) > 1:
            return ", ".join(items[:-1]) + ", dan " + items[-1]
        else:
            return items[0]
    return text

def build_prompt_with_history(history, current_question, documents):
    cleaned_docs = [clean_semicolon_text(doc) for doc in documents]
    context = "\n".join(cleaned_docs)

    # Format riwayat sebagai dialog alami
    history_lines = []
    for turn in history:
        q = turn["question"].strip()
        a = turn["answer"].strip()
        if q and a:
            history_lines.append(f"Pengguna: {q}")
            history_lines.append(f"Asisten: {a}")
    history_text = "\n".join(history_lines)

    # Gabungkan ke dalam prompt
    prompt = f"""
Kamu adalah asisten kesehatan profesional dari alodokter.com.

Gunakan informasi berikut dari artikel kesehatan untuk membantu pengguna secara alami dan menyeluruh.

=== Mulai Konteks Artikel Kesehatan ===
{context}
=== Akhir Konteks ===

Jika informasi yang tersedia tidak cukup untuk menjawab pertanyaan, balas dengan:
"Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

Berikut ini adalah percakapan sebelumnya:
{history_text}

Lanjutkan percakapan di bawah ini.

Pengguna: {current_question}
Asisten:""".strip()

    return prompt


def ask_llama_via_api(prompt):
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        duration = time.time() - start_time

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            return result if result else "⚠️ Model tidak memberikan respons.", duration
        return f"⚠️ Error dari Ollama API: {response.status_code} - {response.text}", duration
    except Exception as e:
        return f"⚠️ Terjadi error saat menghubungi Ollama API: {str(e)}", 0

def main():
    print("🤖 Chatbot Kesehatan Alodokter + LLaMA (Ketik 'exit' untuk keluar)")
    history = []

    while True:
        try:
            question = input("\n🧑‍⚕️ Pertanyaan Anda:  ").strip()
            if question.lower() in {"exit", "quit"}:
                print("👋 Terima kasih, sampai jumpa!")
                break

            docs, metas = query_with_context_expansion(question, top_k=5, window=2)
            valid_docs = [doc.strip() for doc in docs if doc and len(doc.strip()) >= 50]

            if not valid_docs:
                answer = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
                print("\n💬 Jawaban Chatbot:")
                print(answer)
                history.append({"question": question, "answer": answer})
                continue

            prompt = build_prompt_with_history(history, question, valid_docs)
            print("🤔 Sedang berpikir...")
            answer, response_time = ask_llama_via_api(prompt)

            print(f"⏱️  Waktu respons: {response_time:.2f} detik")
            print("\n💬 Jawaban Chatbot:")
            print(answer)
            history.append({"question": question, "answer": answer})

            if metas and "tidak memiliki informasi yang cukup" not in answer.lower():
                print("\n📚 Sumber Terkait:")
                shown = set()
                for meta, doc in zip(metas, docs):
                    href = meta.get('href')
                    if href and href not in shown and doc and len(doc.strip()) >= 50:
                        shown.add(href)
                        print(f"- {meta.get('name')} — {href}")
                        print(f"  📄: {doc.strip()[:150]}...\n")

        except Exception as e:
            print(f"⚠️ Terjadi error: {str(e)}")

if __name__ == "__main__":
    main()
