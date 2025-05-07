import subprocess
import chromadb
import json

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

def chatbot_query(query):
    # Load embeddings database
    client = chromadb.Client()
    collection = client.get_collection("penyakit_embeddings")

    # Mengambil hasil pencocokan yang paling relevan
    results = collection.query(query, n_results=1)

    # Ambil paragraf terdekat dan lakukan query ke Ollama untuk jawabannya
    most_relevant_paragraph = results['documents'][0]
    
    # Gunakan model LLaMA melalui Ollama untuk merespons berdasarkan paragraf yang ditemukan
    prompt = most_relevant_paragraph + " " + query
    
    # Menjalankan Ollama untuk menghasilkan jawaban
    result = subprocess.run(
        ["ollama", "run", "llama3.2", "--text", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Mengambil output dari Ollama yang berisi jawaban
    response = result.stdout.decode("utf-8")
    
    return response

if __name__ == "__main__":
    query = "Apa itu abetalipoproteinemia?"
    response = chatbot_query(query)
    print(f"Response: {response}")
