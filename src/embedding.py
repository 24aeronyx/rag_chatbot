import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Konfigurasi
CHUNKED_FILE = 'Data/penyakit-data-chunked.json'
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

# Load model ke GPU jika tersedia
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

    # Optional: bersihkan collection sebelumnya (jika ingin fresh start)
    # collection.delete()  # Hati-hati, hapus semua data sebelumnya!

    total_chunks = sum(len(entry["chunks"]) for entry in data)
    print(f"\n🧠 Total chunks: {total_chunks}\n")

    all_ids = []
    all_docs = []
    all_embeddings = []
    all_metadatas = []

    for entry in tqdm(data, desc="🔄 Membuat embedding untuk ChromaDB"):
        name = entry["name"]
        href = entry["href"]
        chunks = entry["chunks"]

        for idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_id = f"{href}_{idx}"  # Pastikan unik (pakai href agar beda penyakit tidak bentrok)
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_embeddings.append(get_embedding(chunk))
            all_metadatas.append({
                "name": name,
                "href": href,
                "chunk_index": idx
            })

    # Tambahkan semuanya sekaligus untuk efisiensi
    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings,
        metadatas=all_metadatas
    )

    print(f"\n✅ Selesai menyimpan {len(all_ids)} embedding ke ChromaDB di: {PERSIST_DIR}")

if __name__ == "__main__":
    embed_to_chromadb()
