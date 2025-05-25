import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Konfigurasi
CHUNKED_FILE = 'ata/penyakit-data-chunked.json'
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

# Load model ke GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def get_embedding(text):
    return model.encode(text).tolist()

def embed_to_chromadb():
    # Load chunked data
    with open(CHUNKED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Setup ChromaDB
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = sum(len(entry["chunks"]) for entry in data)
    print(f"\n🧠 Total chunks: {total_chunks}\n")

    counter = 0
    for entry in tqdm(data, desc="🔄 Menyimpan embedding ke ChromaDB"):
        name = entry["name"]
        href = entry["href"]
        chunks = entry["chunks"]

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{name}_{idx}"
            embedding = get_embedding(chunk)

            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "name": name,
                    "href": href,
                    "chunk_index": idx
                }]
            )
            counter += 1

    print(f"\n✅ Selesai menyimpan {counter} embedding ke ChromaDB di: {PERSIST_DIR}")

if __name__ == "__main__":
    embed_to_chromadb()
