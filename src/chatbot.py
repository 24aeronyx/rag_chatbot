import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import sys

# Inisialisasi model embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Konfigurasi ChromaDB
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

# Setup Chroma Client dan Collection
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

def query(question, top_k=3):
    # Embedding untuk pertanyaan
    embedding = model.encode(question).tolist()

    # Query ke ChromaDB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    # Ambil hasil teratas
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Tampilkan hasil
    print("\n📚 Hasil yang ditemukan:\n" + "-" * 50)
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        print(f"\n🔎 Hasil {i}")
        print(f"📝 Penyakit  : {meta['name']}")
        print(f"🔗 Link      : {meta['href']}")
        print(f"📄 Potongan  : {doc}")

def main():
    print("🤖 Chatbot Kesehatan Alodokter (Ketik 'exit' untuk keluar)")
    while True:
        user_input = input("\n🧑‍⚕️ Pertanyaan Anda: ")
        if user_input.strip().lower() in {'exit', 'quit'}:
            print("👋 Terima kasih, sampai jumpa!")
            break
        query(user_input)

if __name__ == "__main__":
    main()
