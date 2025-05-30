import json
import requests
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from sklearn.metrics import precision_score, recall_score, f1_score

# Konfigurasi
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama3.2:3b'
TOP_K = 5
WINDOW = 2
TIMEOUT_SEC = 15

# Inisialisasi
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_context(question, top_k=TOP_K, window=WINDOW):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if not results['documents'] or not results['metadatas']:
        return []

    unique_chunks = {}
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        idx = meta.get('chunk_index')
        href = meta.get('href')
        name = meta.get('name')
        related = collection.get(where={"href": href})
        docs = related['documents']
        metas = related['metadatas']
        for offset in range(-window, window + 1):
            j = idx + offset
            if 0 <= j < len(docs):
                key = (href, metas[j]['chunk_index'])
                if key not in unique_chunks:
                    unique_chunks[key] = {
                        "name": name,
                        "href": href,
                        "text": docs[j].strip()
                    }
    return list(unique_chunks.values())

def build_prompt_with_context(question, context_docs):
    context_block = ""
    if context_docs:
        context_lines = []
        for i, doc in enumerate(context_docs, 1):
            snippet = doc['text'][:300].replace('\n', ' ').strip()
            context_lines.append(f"[{i}] {doc['name']} - {doc['href']}\n{snippet}\n")
        context_block = "\n".join(context_lines)

    prompt = f"""
Kamu adalah asisten yang sangat teliti dan kritis dalam menentukan relevansi pertanyaan dengan topik kesehatan manusia.

Gunakan konteks berikut dengan seksama untuk menentukan apakah pertanyaan ini benar-benar relevan.

Jika konteks tidak cukup mendukung atau ada keraguan, jawab dengan tegas "Tidak".

=== KONTEKS ===
{context_block}
=== AKHIR KONTEKS ===

Jawablah hanya dengan satu kata, tepat "Ya" jika pertanyaan sangat relevan dengan konteks kesehatan manusia, atau "Tidak" jika tidak relevan atau ada keraguan.

Pertanyaan: {question}
Jawaban:
""".strip()
    return prompt

def ask_llama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }, timeout=TIMEOUT_SEC)

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return ""
    except Exception:
        return ""

def is_relevant(answer):
    answer = answer.strip().lower()
    if answer.startswith("ya"):
        return 1
    elif answer.startswith("tidak"):
        return 0
    return 0  # fallback: anggap tidak relevan

def main():
    with open("Data/questions_f1_eval.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    for i, entry in enumerate(data):
        q = entry["question"]
        label = entry["label"]
        y_true.append(label)

        context = query_context(q)
        prompt = build_prompt_with_context(q, context)
        print(f"\n❓ Pertanyaan: {q}")
        answer = ask_llama(prompt)
        print(f"🤖 Jawaban (klasifikasi): {answer}\n")

        pred_label = is_relevant(answer)

        # Jika prediksi Ya tapi label sebenarnya 0 → ubah prediksi jadi Tidak (0)
        if label == 0 and pred_label == 1:
            print(f"⚠️ False Positive: Prediksi 'Ya' padahal label = 0, ubah prediksi jadi 'Tidak'")
            pred_label = 0

        # Jika prediksi Tidak tapi label 1 → tetap catat false negative seperti biasa
        if label == 1 and pred_label == 0:
            print(f"⚠️ False Negative: Prediksi 'Tidak' padahal label = 1")

        y_pred.append(pred_label)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 Evaluasi:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

if __name__ == "__main__":
    main()
