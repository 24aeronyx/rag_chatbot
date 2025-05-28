import requests
import json
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

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
{chr(10).join(sample['ground_truths'])}

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

    # Ollama response biasanya berisi kunci 'response' dengan teks output
    raw_output = data.get("response", "")

    try:
        # parsing JSON output langsung
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        # Jika gagal parse, coba cari JSON di dalam teks menggunakan trik sederhana
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

def save_results_to_csv(results, filename="evaluation_results.csv"):
    if not results:
        print("No results to save.")
        return
    keys = results[0].keys()
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {filename}")

def main():
    # Contoh data sample evaluasi
    samples = [
        {
            "question": "Apa itu flu burung?",
            "answer": "Flu burung adalah penyakit yang disebabkan oleh virus influenza tipe A yang biasanya menular dari unggas ke manusia.",
            "contexts": [
                "Flu burung adalah penyakit yang disebabkan oleh virus influenza tipe A.",
                "Virus ini biasanya menular dari unggas ke manusia."
            ],
            "ground_truths": [
                "Flu burung disebabkan virus influenza tipe A yang menyerang unggas.",
                "Penyakit ini bisa menular ke manusia melalui kontak dengan unggas yang terinfeksi."
            ],
        },
        {
            "question": "Bagaimana cara mengobati Malaria?",
            "answer": "Mengobati malaria dengan obat antimalaria seperti klorokuin dan menghindari gigitan nyamuk.",
            "contexts": [
                "Malaria adalah penyakit infeksi yang ditularkan oleh nyamuk Anopheles.",
                "Pengobatan utama malaria adalah dengan obat antimalaria."
            ],
            "ground_truths": [
                "Malaria harus segera ditangani dengan obat antimalaria.",
                "Pencegahan termasuk menghindari gigitan nyamuk Anopheles."
            ],
        },
    ]

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"Evaluating sample #{i}...")
        result = evaluate_sample(sample)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("Evaluation result:", result)
        # Gabungkan hasil dengan sample info untuk simpan CSV
        record = {
            "index": i,
            "question": sample["question"],
            "answer": sample["answer"],
            **result
        }
        results.append(record)

    save_results_to_csv(results)

if __name__ == "__main__":
    main()
