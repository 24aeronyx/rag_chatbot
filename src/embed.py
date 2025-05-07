import json
import chromadb
import subprocess
import numpy as np

def get_embedding(text):
    # Mengirimkan prompt untuk memperoleh embedding dari Ollama LLaMA 3.2
    result = subprocess.run(
        ["ollama", "run", "llama3.2", "--embed", text],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Mengambil output JSON dari Ollama
    embedding = json.loads(result.stdout)
    
    return embedding['embedding']

def embed_data(input_file, output_file):
    # Load data JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Setup chromadb client
    client = chromadb.Client()

    # Buat collection baru untuk embeddings
    collection = client.create_collection(name="penyakit_embeddings")

    # Embedding setiap paragraf
    for entry in data:
        for paragraph in entry['paragraphs']:
            # Menggunakan Ollama untuk mendapatkan embedding
            embedding = get_embedding(paragraph)

            # Simpan embedding ke database
            collection.add(
                documents=[paragraph],
                embeddings=[embedding],
                metadatas=[{"name": entry['name'], "href": entry['href']}]
            )

    # Simpan database embedding
    collection.save(output_file)

if __name__ == "__main__":
    embed_data('./data/penyakit-data-processed.json', './embeddings/penyakit-embeddings.db')
