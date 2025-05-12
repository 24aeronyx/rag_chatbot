import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import sys

# Setup
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Init ChromaDB & model embedding
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_to_chroma(question, top_k=5):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    return documents, metadatas

def build_prompt(question, documents):
    context = "\n".join(documents)
    prompt = f"""
Berikut ini adalah informasi dari berbagai artikel kesehatan:

{context}

Berdasarkan informasi di atas, jawablah pertanyaan berikut secara jelas dan informatif:

Pertanyaan: {question}

Jawaban:"""
    return prompt.strip()

def ask_llama_via_api(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"⚠️ Error dari Ollama API: {response.status_code} - {response.text}"

def main():
    print("🤖 Chatbot Kesehatan Alodokter + LLaMA (Ketik 'exit' untuk keluar)")
    while True:
        question = input("\n🧑‍⚕️ Pertanyaan Anda: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Terima kasih, sampai jumpa!")
            break

        docs, metas = query_to_chroma(question)
        prompt = build_prompt(question, docs)
        answer = ask_llama_via_api(prompt)

        print("\n💬 Jawaban Chatbot:")
        print(answer)

        print("\n📚 Sumber Terkait:")
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            print(f"\n{i}. {meta['name']} — {meta['href']}")
            print(f"   📄: {doc[:100]}...")  # tampilkan preview isi

if __name__ == "__main__":
    main()
