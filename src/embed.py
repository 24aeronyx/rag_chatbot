import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Konfigurasi
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHECKPOINT_FILE = './embeddings/checkpoint-embedding.json'
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    return model.encode(text).tolist()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": 0, "last_paragraph_index": 0, "last_chunk_index": 0}

def save_checkpoint(progress):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f)

def embed_data(input_file):
    # Load data JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Setup ChromaDB Persistent Client
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed = checkpoint['processed']
    last_paragraph_index = checkpoint['last_paragraph_index']
    last_chunk_index = checkpoint['last_chunk_index']

    print(f"\n🧠 Melanjutkan dari penyakit ke-{processed}...\n")

    for disease_index, entry in enumerate(tqdm(data[processed:], desc="🔄 Embedding data penyakit", initial=processed, total=len(data))):
        name = entry["name"]
        href = entry["href"]
        paragraphs = entry["paragraphs"]

        for i, paragraph in enumerate(paragraphs[last_paragraph_index:] if disease_index == 0 else paragraphs):
            chunks = chunk_text(paragraph)

            for j, chunk in enumerate(chunks[last_chunk_index:] if disease_index == 0 and i == last_paragraph_index else chunks):
                chunk_id = f"{name}_{i}_{j}"
                embedding = get_embedding(chunk)

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "name": name,
                        "href": href,
                        "paragraph_index": i,
                        "chunk_index": j
                    }]
                )

                # Update checkpoint
                checkpoint = {
                    "processed": processed + disease_index,
                    "last_paragraph_index": i,
                    "last_chunk_index": j + 1
                }
                save_checkpoint(checkpoint)

            # Reset chunk index setiap paragraf baru
            last_chunk_index = 0

        # Reset paragraph index setiap penyakit baru
        last_paragraph_index = 0

    print("\n✅ Embedding selesai dan disimpan ke ChromaDB.")

if __name__ == "__main__":
    embed_data('./data/penyakit-data-processed.json')
