import json
import asyncio
from pprint import pprint
from typing import List
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings


# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
TOP_K = 5
QUESTIONS_FILE = 'Data/generated-questions.json'
OUTPUT_CSV = 'Data/ragas-eval.csv'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2:3b'
TIMEOUT_SEC = 10
MAX_QUESTIONS = 25


# --- Wrapper LLM untuk Ollama ---
class OllamaLLMWrapper(BaseRagasLLM):
    def __init__(self, url, model_name='llama3.2:3b', timeout=10):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout

    def generate_text(self, prompt: str) -> str:
        import requests
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"⚠️ Error {response.status_code}: {response.text}"
        except requests.exceptions.Timeout:
            return "⚠️ Request ke Ollama API timeout."
        except Exception as e:
            return f"⚠️ Terjadi error: {str(e)}"

    async def agenerate_text(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        # Pastikan ini return await (bukan hanya return)
        return await loop.run_in_executor(None, self.generate_text, prompt)


# --- Wrapper Embeddings untuk SentenceTransformer ---
class SentenceTransformerEmbedder(BaseRagasEmbeddings):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=False)

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], convert_to_numpy=False)[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, query)


# --- Inisialisasi ---
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformerEmbedder()
llm = OllamaLLMWrapper(url=OLLAMA_URL, model_name=MODEL_NAME, timeout=TIMEOUT_SEC)


# --- Ambil konteks dari ChromaDB dengan embedding dan k-nearest ---
def get_contexts(question: str, top_k=TOP_K):
    embedding = embedder.embed_query(question)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # Filter dan ambil teks dokumen yang cukup panjang
    contexts = []
    for doc, meta in zip(docs, metas):
        if doc and len(doc.strip()) > 50:
            contexts.append(doc.strip())
    return contexts


# --- Prompt builder ---
def build_prompt(contexts: List[str], question: str) -> str:
    context_str = "\n".join([f"- {ctx[:300].replace('\n', ' ').strip()}" for ctx in contexts])
    return f"""
Kamu adalah asisten kesehatan profesional. Berdasarkan informasi berikut, bantu jawab pertanyaan pengguna secara langsung dan profesional.

=== Informasi milikmu ===
{context_str}
=== Akhir Informasi ===

Pertanyaan pengguna: "{question}"
Jawaban:
""".strip()


# --- Buat EvaluationDataset dari pertanyaan dan jawaban model ---
def generate_ragas_dataset(questions, max_questions=MAX_QUESTIONS):
    ragas_data = []

    for item in tqdm(questions[:max_questions], desc="⏳ Membuat dataset RAGAS"):
        question = item.get('question', '').strip()
        if not question:
            continue

        contexts = get_contexts(question)
        if not contexts:
            continue

        prompt = build_prompt(contexts, question)
        answer = llm.generate_text(prompt)
        if not answer or answer.startswith("⚠️"):
            continue

        ragas_data.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
        })

    return EvaluationDataset(ragas_data)


# --- Eksekusi utama ---
if __name__ == "__main__":
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    dataset = generate_ragas_dataset(questions)

    print("\n🚀 Menjalankan evaluasi RAGAS dengan LLaMA Ollama...")

    result = evaluate(
        dataset=dataset,
        llm=llm,
        embeddings=embedder,
        metrics=[faithfulness, answer_relevancy],
    )

    df = result.to_pandas()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Evaluasi selesai. Hasil disimpan ke: {OUTPUT_CSV}")
    print(df.head())
