import json
import asyncio
import requests
from ragas import EvaluationDataset, evaluate
from ragas.metrics import SemanticSimilarity

# Wrapper async untuk memanggil LLM Ollama API
class CustomLLMWrapper:
    def __init__(self, base_url, model, max_tokens=8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
    
    async def generate(self, prompt, **kwargs):
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens)
        }
        
        # Gunakan asyncio.to_thread untuk menjalankan requests.post secara async
        response = await asyncio.to_thread(
            requests.post, f"{self.base_url}/generate", headers=headers, json=payload
        )
        
        if response.status_code == 200:
            # Perhatikan ini, ambil dari key 'response'
            return response.json().get("response", "").strip()
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

# Wrapper async untuk memanggil embedding model
class CustomEmbeddingsWrapper:
    def __init__(self, base_url, model):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed_text(self, text):
        headers = {
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}
        
        response = await asyncio.to_thread(
            requests.post, f"{self.base_url}/embeddings", headers=headers, json=payload
        )
        
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

# Fungsi membuat prompt dari question dan konteks
def build_prompt(question, contexts):
    context_text = "\n\n".join([f"{c['name']} ({c['href']}): {c['text'][:300]}..." for c in contexts])
    prompt = f"""\
Kamu adalah asisten kesehatan profesional. Berdasarkan konteks berikut, jawab pertanyaan ini secara ringkas dan jelas:

{context_text}

Pertanyaan: {question}

Jawaban:"""
    return prompt

# Generate jawaban dari LLM untuk tiap pertanyaan di dataset
async def generate_answers(dataset, llm):
    results = []
    for entry in dataset:
        question = entry['question']
        contexts = entry.get('contexts', [])
        prompt = build_prompt(question, contexts)
        answer = await llm.generate(prompt)
        results.append({
            "question": question,
            "contexts": contexts,
            "reference_answer": entry.get('answer', ''),
            "generated_answer": answer
        })
    return results

async def main():
    # Load dataset JSON (format list of dict)
    with open("Data/ragas_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Setup base URL Ollama API, nama model
    base_url = "http://localhost:11434/api"   # Sesuaikan endpointmu
    llm_model = "llama3.2:3b"
    embedding_model = "all-MiniLM-L6-v2"       # Pastikan model embedding ini tersedia di API embedding-mu

    llm = CustomLLMWrapper(base_url=base_url, model=llm_model)
    embedder = CustomEmbeddingsWrapper(base_url=base_url, model=embedding_model)

    # Generate jawaban untuk tiap pertanyaan di dataset
    generated_results = await generate_answers(raw_data, llm)

    eval_dataset = EvaluationDataset.from_list([
    {
        "user_input": item["question"],
        "contexts": item["contexts"],            # konteks referensi atau ground truth
        "retrieved_contexts": item["contexts"],  # konteks yang dipakai model (bisa sama dengan contexts)
        "response": item["generated_answer"],
        "reference": ""  # kosong jika tidak ada jawaban referensi
    }
    for item in generated_results
])

    # Metric evaluasi berbasis similarity embedding
    metrics = [SemanticSimilarity(embeddings=embedder)]

    results = evaluate(dataset=eval_dataset)
    print(results)
    df = results.to_pandas()
    df.to_csv("ragas_evaluation.csv", index=False)
    print("✅ Evaluasi selesai dan disimpan ke ragas_evaluation.csv")

if __name__ == "__main__":
    asyncio.run(main())
