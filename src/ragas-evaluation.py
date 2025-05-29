import requests
import json
import csv
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
INPUT_FILE = "Data/ragas-dataset.json"
OUTPUT_FILE = "Data/ragas_results.csv"

def create_prompt(sample):
    prompt = f"""
Anda adalah evaluator jawaban chatbot berbasis LLM. Tugas Anda adalah mengevaluasi kualitas jawaban berdasarkan konteks dan referensi yang tersedia. Nilai diberikan dalam rentang 0 sampai 1, menggunakan empat metrik evaluasi. Jangan beri penjelasan tambahan apa pun. Berikan jawaban akhir dalam format JSON.

### Metrik dan Definisi:

1. **faithfulness**: Apakah klaim-klaim dalam jawaban sepenuhnya didukung oleh konteks?
   Rumus:
   faithfulness = jumlah klaim yang didukung konteks / total klaim dalam jawaban

2. **answer_relevance**: Apakah jawaban relevan terhadap pertanyaan?
   Rumus (disimulasikan melalui penilaian semantik):
   answer_relevance = rata-rata kesamaan semantik antara pertanyaan asli dan pertanyaan buatan dari jawaban

3. **context_precision**: Apakah potongan konteks yang diberikan relevan dan benar?
   Rumus:
   precision@k = true positives / (true positives + false positives)
   context_precision = rerata dari precision@k untuk setiap chunk relevan

4. **context_recall**: Apakah konteks mencakup informasi penting dari referensi?
   Rumus:
   context_recall = jumlah klaim dalam referensi yang didukung oleh konteks / total klaim dalam referensi

### Data untuk Evaluasi:

Pertanyaan:
{sample['question']}

Jawaban model:
{sample['answer']}

Konteks yang digunakan:
{chr(10).join(sample['contexts'])}

Referensi kebenaran (ground truth):
{chr(10).join([ref for ref in sample['ground_truths'] if ref.strip()])}

### Format keluaran yang diminta:
Hanya jawab dengan format JSON berikut, tanpa penjelasan tambahan:

{{
  "context_precision": float,
  "context_recall": float,
  "answer_relevance": float,
  "faithfulness": float
}}
"""
    return prompt.strip()

def evaluate_sample(sample):
    prompt = create_prompt(sample)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"HTTP request error: {str(e)}"}

    data = response.json()
    raw_output = data.get("response", "")

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {"error": "Failed to parse JSON from model output", "raw_output": raw_output}
        else:
            result = {"error": "No JSON found in model output", "raw_output": raw_output}

    return result

def save_results_to_csv(results, mean_scores, filename=OUTPUT_FILE):
    if not results:
        print("No results to save.")
        return

    keys = [
        "question",
        "answer",
        "contexts",
        "ground_truths",
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    ]

    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in keys})

        # Baris tambahan untuk mean scores
        writer.writerow({
            "question": "MEAN SCORES",
            "faithfulness": round(mean_scores["faithfulness"], 4),
            "answer_relevance": round(mean_scores["answer_relevance"], 4),
            "context_precision": round(mean_scores["context_precision"], 4),
            "context_recall": round(mean_scores["context_recall"], 4),
        })

    print(f"Saved results to {filename}")

def load_samples_from_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def main():
    samples = load_samples_from_json(INPUT_FILE)
    if not samples:
        print("No data to evaluate.")
        return

    results = []
    sum_faithfulness = sum_answer_relevance = sum_context_precision = sum_context_recall = 0
    count = 0

    for i, sample in enumerate(samples, 1):
        print(f"Evaluating sample #{i}...")
        result = evaluate_sample(sample)

        if "error" in result:
            print(f"Error: {result['error']}")
            faithfulness = answer_relevance = context_precision = context_recall = None
        else:
            faithfulness = result.get("faithfulness", None)
            answer_relevance = result.get("answer_relevance", None)
            context_precision = result.get("context_precision", None)
            context_recall = result.get("context_recall", None)

            if None not in (faithfulness, answer_relevance, context_precision, context_recall):
                sum_faithfulness += faithfulness
                sum_answer_relevance += answer_relevance
                sum_context_precision += context_precision
                sum_context_recall += context_recall
                count += 1

        record = {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": json.dumps(sample["contexts"], ensure_ascii=False),
            "ground_truths": json.dumps(sample["ground_truths"], ensure_ascii=False),
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        results.append(record)

    if count > 0:
        mean_scores = {
            "faithfulness": sum_faithfulness / count,
            "answer_relevance": sum_answer_relevance / count,
            "context_precision": sum_context_precision / count,
            "context_recall": sum_context_recall / count,
        }

        print("\n==== MEAN SCORES ====")
        for k, v in mean_scores.items():
            print(f"{k.capitalize().replace('_', ' ')}: {round(v, 4)}")

    else:
        mean_scores = {k: 0.0 for k in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]}
        print("Tidak ada skor yang valid untuk dihitung rata-ratanya.")

    save_results_to_csv(results, mean_scores)

if __name__ == "__main__":
    main()
