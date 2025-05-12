import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import sys

# Inisialisasi embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Konfigurasi ChromaDB Persistent Client
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

# Inisialisasi Chroma
client = PersistentClient(path=PERSIST_DIR)

# Ambil koleksi embedding
collection = client.get_or_create_collection(COLLECTION_NAME)

def query(question, top_k=3):
    # Encode pertanyaan
    embedding = model.encode(question).tolist()

    # Query ke ChromaDB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    # Ambil dokumen hasil
    responses = results['documents'][0]
    metadatas = results['metadatas'][0]

    # Format jawaban
    print("\n📚 Hasil yang ditemukan:")
    for i, (doc, meta) in enumerate(zip(responses, metadatas)):
        print(f"\nResult {i+1}:")
        print(f"📝 Penyakit: {meta['name']}")
        print(f"🔗 Link: {meta['href']}")
        print(f"📄 Konten: {doc}")

if __name__ == "__main__":
    print("🤖 Chatbot Kesehatan (Ketik 'exit' untuk keluar)")
    while True:
        user_input = input("\n🧑‍⚕️ Pertanyaan Anda: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Sampai jumpa!")
            sys.exit()
        query(user_input)
