import asyncio
import csv
import requests
import time

from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.dataset_schema import SingleTurnSample

from sentence_transformers import SentenceTransformer
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    Generation,
    LLMResult,
)


# Konfigurasi endpoint dan model Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


# Custom Langchain LLM client untuk Ollama API
class OllamaAPIClient(BaseChatModel):
    model_name: str = MODEL_NAME
    base_url: str = OLLAMA_URL

    def _call(self, messages, stop=None) -> str:
        # Gabungkan isi pesan sebagai prompt teks
        prompt_parts = [m.content for m in messages]
        prompt = "\n".join(prompt_parts)

        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            # Parsing response sesuai format Ollama API
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if content:
                return content
            else:
                return "⚠️ Model tidak memberikan respons."
        except Exception as e:
            return f"⚠️ Error saat menghubungi Ollama API: {str(e)}"

    async def _acall(self, messages, stop=None) -> str:
        return await asyncio.to_thread(self._call, messages, stop)

    @property
    def _llm_type(self) -> str:
        return "ollama_api"

    def _generate(self, messages, stop=None):
        text = self._call(messages, stop)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]], llm_output={})

    async def _agenerate(self, messages, stop=None):
        text = await self._acall(messages, stop)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]], llm_output={})


async def evaluate_samples(samples):
    # Embedding (kalau kamu perlu untuk indexing/penyiapan, tapi TIDAK dipakai di konstruktor metrik)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Inisialisasi LLM client Ollama
    llm = OllamaAPIClient()

    # Inisialisasi metrik RAGAS, tanpa embedding argument
    context_precision = LLMContextPrecisionWithReference(llm=llm)
    context_recall = LLMContextRecall(llm=llm)
    answer_relevance = AnswerRelevancy(llm=llm)
    faithfulness = Faithfulness(llm=llm)

    results = []

    for idx, sample in enumerate(samples, start=1):
        print(f"Evaluating sample #{idx}...")

        cp = await context_precision.single_turn_ascore(sample)
        cr = await context_recall.single_turn_ascore(sample)
        ar = await answer_relevance.single_turn_ascore(sample)
        fh = await faithfulness.single_turn_ascore(sample)

        results.append({
            "index": idx,
            "user_input": sample.user_input,
            "response": sample.response,
            "context_precision": cp,
            "context_recall": cr,
            "answer_relevance": ar,
            "faithfulness": fh,
        })

    return results


def save_results_to_csv(results, filepath="evaluation_results.csv"):
    keys = results[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved evaluation results to {filepath}")


async def main():
    # Contoh data evaluasi, ganti sesuai datasetmu
    samples = [
        SingleTurnSample(
            user_input="Apa itu flu burung?",
            response="Flu burung adalah penyakit menular pada unggas...",
            retrieved_contexts=[
                "Flu burung adalah penyakit yang disebabkan oleh virus influenza tipe A...",
                "Virus ini biasanya menular dari unggas ke manusia..."
            ],
            reference_contexts=[
                "Flu burung adalah penyakit yang disebabkan virus influenza tipe A...",
                "Penyakit ini dapat menular dari unggas ke manusia..."
            ],
        ),
        # Tambahkan samples lain bila perlu
    ]

    results = await evaluate_samples(samples)
    save_results_to_csv(results)


if __name__ == "__main__":
    asyncio.run(main())
