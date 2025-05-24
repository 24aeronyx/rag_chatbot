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

def query_to_chroma(question, top_k=3):  
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
    start_time = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    end_time = time.time()
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

        docs, metas = query_to_chroma(question, top_k=3)
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

            shown_sources = {}
            for meta, doc in zip(metas, docs):
                href = meta.get('href')
                if href and href not in shown_sources and doc and len(doc.strip()) >= 50:
                    shown_sources[href] = {
                        "name": meta.get("name"),
                        "href": href,
                        "snippet": doc.strip()[:150]
                    }

            for i, (href, info) in enumerate(shown_sources.items(), start=1):
                print(f"{i}. {info['name']} — {info['href']}")
                print(f"   📄: {info['snippet']}...\n")


if __name__ == "__main__":
    main()
