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
MODEL_NAME = "llama3.2"

# Init ChromaDB & model embedding
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_to_chroma(question, top_k=1):  
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    return documents, metadatas

def build_prompt_with_history(history, current_question, documents):
    context = "\n".join(documents)
    history_text = ""
    for turn in history:
        history_text += f"Pengguna: {turn['question']}\nAsisten: {turn['answer']}\n"
    history_text += f"Pengguna: {current_question}\nAsisten:"

    prompt = f"""
Kamu adalah asisten kesehatan profesional yang hanya boleh menjawab pertanyaan berdasarkan informasi yang terdapat dalam konteks di bawah ini.

=== Mulai Konteks Artikel Kesehatan ===
{context}
=== Akhir Konteks ===

Jika informasi yang diberikan tidak cukup untuk menjawab pertanyaan, jawab dengan:
"Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

{history_text}
    """.strip()

    return prompt


def ask_llama_via_api(prompt):
    start_time = time.time()  # ⏱️ Start timer

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    end_time = time.time()  # ⏱️ End timer
    duration = end_time - start_time

    if response.status_code == 200:
        return response.json()["response"], duration
    else:
        return f"⚠️ Error dari Ollama API: {response.status_code} - {response.text}", duration

def main():
    print("🤖 Chatbot Kesehatan Alodokter + LLaMA (Ketik 'exit' untuk keluar)")
    history = []

    while True:
        question = input("\n🧑‍⚕️ Pertanyaan Anda: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Terima kasih, sampai jumpa!")
            break

        docs, metas = query_to_chroma(question, top_k=1)

        # Cek dokumen utama (dokumen tunggal)
        doc = docs[0].strip() if docs else ""
        meta = metas[0] if metas else None

        # Jika dokumen kosong atau sangat pendek, anggap tidak relevan
        if len(doc) < 50:  # threshold bisa disesuaikan
            answer = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
            print("\n💬 Jawaban Chatbot:")
            print(answer)
            history.append({"question": question, "answer": answer})
            continue

        prompt = build_prompt_with_history(history, question, [doc])

        print("🤔 Sedang berpikir...")
        answer, response_time = ask_llama_via_api(prompt)
        print(f"⏱️  Waktu respons: {response_time:.2f} detik")
        print("\n💬 Jawaban Chatbot:")
        print(answer)

        history.append({
            "question": question,
            "answer": answer
        })

        if meta and "tidak memiliki informasi yang cukup" not in answer.lower():
            print("\n📚 Sumber Terkait:")
            print(f"1. {meta['name']} — {meta['href']}")
            print(f"   📄: {doc[:150]}...")


if __name__ == "__main__":
    main()
