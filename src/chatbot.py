import os
import json
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2:3b'
TOP_K = 5
WINDOW = 2
TIMEOUT_SEC = 15
HISTORY_DIR = 'history'
MAX_HISTORY = 3

# Inisialisasi
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

history = []
history_filepath = None

def save_history():
    if history_filepath and history:
        with open(history_filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

def create_history_file():
    global history_filepath
    os.makedirs(HISTORY_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'history_{timestamp}.json'
    history_filepath = os.path.join(HISTORY_DIR, filename)

def query_context_with_history(question, recent_history, top_k=TOP_K, window=WINDOW):
    texts_to_embed = [turn['question'] for turn in recent_history[-2:]] + [question]
    combined_text = " ".join(texts_to_embed)

    embedding = embedder.encode(combined_text).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if not results['documents'] or not results['metadatas']:
        return []

    unique_chunks = {}
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        idx = meta.get('chunk_index')
        href = meta.get('href')
        name = meta.get('name')

        related = collection.get(where={"href": href})
        docs = related['documents']
        metas = related['metadatas']

        for offset in range(-window, window + 1):
            j = idx + offset
            if 0 <= j < len(docs):
                key = (href, metas[j]['chunk_index'])
                if key not in unique_chunks:
                    unique_chunks[key] = {
                        "name": name,
                        "href": href,
                        "text": docs[j].strip()
                    }
    return list(unique_chunks.values())


def build_prompt(context_docs, question, recent_history):
    if not context_docs:
        question_clean = ''.join(c for c in question if c.isalnum() or c.isspace()).strip()
        return f"Maaf, saya tidak memiliki informasi yang cukup tentang {question_clean}"

    # Buat blok konteks
    context_lines = []
    for i, doc in enumerate(context_docs, 1):
        snippet = doc['text'][:300].replace('\n', ' ').strip()
        context_lines.append(f"[{i}] {doc['name']} - {doc['href']}\n{snippet}\n")
    context_block = "\n".join(context_lines)

    # Buat dialog history natural
    history_lines = []
    for turn in recent_history:
        q = turn.get('question', '').strip()
        a = turn.get('answer', '').strip()
        history_lines.append(f"User: {q}\nAssistant: {a}")
    history_block = "\n".join(history_lines)

    prompt = f"""
Kamu adalah asisten kesehatan profesional yang ramah dan jelas. Berdasarkan informasi berikut, bantu jawab pertanyaan pengguna secara lengkap dan profesional. Gunakan konteks dan riwayat percakapan sebelumnya untuk menjaga kesinambungan dialog.

=== Informasi yang kamu miliki ===
{context_block}
=== Akhir Informasi ===

=== Riwayat Percakapan ===
{history_block}
=== Akhir Riwayat ===

User: {question}
Assistant:
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
            return response.json().get("response", "").strip()
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Gagal menghubungi LLaMA: {str(e)}"

def show_references(context_docs):
    # Filter supaya setiap href hanya muncul sekali
    seen = set()
    lines = []
    for doc in context_docs:
        if doc['href'] not in seen:
            seen.add(doc['href'])
            lines.append(f"{doc['name']} - {doc['href']}")
    return "\n".join(lines) if lines else "Tidak ada referensi."

def start_chat():
    print("\n🩺 Chatbot Kesehatan Alodokter")
    print("Ketik 'exit' untuk keluar.\n")

    create_history_file()

    while True:
        question = input("❓ Pertanyaan: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("👋 Sampai jumpa!\n")
            save_history()
            break

        if not question:
            continue

        print("🤖 Sedang mencari jawaban...\n")
        context = query_context_with_history(question, history[-MAX_HISTORY:])
        prompt = build_prompt(context, question, history[-MAX_HISTORY:])

        if prompt.startswith("Maaf, saya tidak memiliki informasi"):
            answer = prompt
        else:
            answer = ask_llama(prompt)

        print("🤖 Jawaban:\n" + answer + "\n")

        if context:
            print("📚 Referensi:")
            print(show_references(context))
            print()

        history.append({"question": question, "answer": answer})
        save_history()

if __name__ == "__main__":
    start_chat()