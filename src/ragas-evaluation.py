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
Anda adalah evaluator jawaban chatbot. Tugas Anda adalah memberikan 4 skor antara 0 sampai 1 dalam format JSON tanpa penjelasan tambahan:

1. context_precision: Seberapa relevan dan benar konteks yang digunakan dalam jawaban.
2. context_recall: Seberapa lengkap konteks tersebut mencakup informasi penting dari referensi.
3. answer_relevance: Seberapa tepat jawaban menjawab pertanyaan.
4. faithfulness: Seberapa sesuai jawaban dengan fakta dalam konteks dan referensi.

Berikut data yang akan Anda evaluasi:

Pertanyaan:
{sample['question']}

Jawaban model:
{sample['answer']}

Konteks yang diambil:
{chr(10).join(sample['contexts'])}

Referensi kebenaran:
{chr(10).join([ref for ref in sample['ground_truths'] if ref.strip()])}

Berikan hasil dalam format JSON persis seperti ini (tanpa tambahan teks atau penjelasan):

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

def save_results_to_csv(results, filename=OUTPUT_FILE):
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
        "mean_score",
    ]

    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            ordered_row = {key: row.get(key, "") for key in keys}
            writer.writerow(ordered_row)

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
    for i, sample in enumerate(samples, 1):
        print(f"Evaluating sample #{i}...")
        result = evaluate_sample(sample)

        if "error" in result:
            print(f"Error: {result['error']}")
            faithfulness = answer_relevance = context_precision = context_recall = mean_score = None
        else:
            faithfulness = result.get("faithfulness", None)
            answer_relevance = result.get("answer_relevance", None)
            context_precision = result.get("context_precision", None)
            context_recall = result.get("context_recall", None)

            if None not in (faithfulness, answer_relevance, context_precision, context_recall):
                mean_score = round(
                    (faithfulness + answer_relevance + context_precision + context_recall) / 4, 4
                )
            else:
                mean_score = None

        record = {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": json.dumps(sample["contexts"], ensure_ascii=False),
            "ground_truths": json.dumps(sample["ground_truths"], ensure_ascii=False),
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "mean_score": mean_score,
        }
        results.append(record)

    save_results_to_csv(results)


if __name__ == "__main__":
    main()
