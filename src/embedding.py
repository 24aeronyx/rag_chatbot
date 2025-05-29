import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Konfigurasi
CHUNKED_FILE = 'Data/penyakit-data-chunked.json'
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
BATCH_SIZE = 1000  # Aman & efisien

# Load model ke GPU jika tersedia
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def get_embedding(text):
    return model.encode(text).tolist()

def embed_to_chromadb():
    with open(CHUNKED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = sum(len(entry["chunks"]) for entry in data)
    print(f"\n🧠 Total chunks: {total_chunks}\n")

    batch_ids = []
    batch_docs = []
    batch_embeddings = []
    batch_metadatas = []
    total_added = 0

    for entry in tqdm(data, desc="🔄 Membuat embedding untuk ChromaDB"):
        name = entry["name"]
        href = entry["href"]
        chunks = entry["chunks"]

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_id = f"{href}_{idx}"
            embedding = get_embedding(chunk)

            batch_ids.append(chunk_id)
            batch_docs.append(chunk)
            batch_embeddings.append(embedding)
            batch_metadatas.append({
                "name": name,
                "href": href,
                "chunk_index": idx
            })

            if len(batch_ids) >= BATCH_SIZE:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                batch_ids, batch_docs, batch_embeddings, batch_metadatas = [], [], [], []

    # Tambahkan sisa data yang belum masuk batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas
        )
        total_added += len(batch_ids)

    print(f"\n✅ Selesai menyimpan {total_added} embedding ke ChromaDB di: {PERSIST_DIR}")

if __name__ == "__main__":
    embed_to_chromadb()
