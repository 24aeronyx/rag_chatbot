import json
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Setup
PERSIST_DIR = './embeddings'
COLLECTION_NAME = 'penyakit_embeddings'
TOP_K = 5
QUESTIONS_FILE = 'data/generated-questions.json'

# Init
client = PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def query_to_chroma(question, top_k=TOP_K):
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results['metadatas'][0]

def compute_metrics(questions):
    average_precisions = []
    reciprocal_ranks = []

    for item in tqdm(questions):
        q = item['question']
        gt = item['ground_truth'].lower()

        metadatas = query_to_chroma(q)
        names = [m['name'].lower() for m in metadatas]

        # Calculate MRR (Mean Reciprocal Rank)
        rank = next((i for i, name in enumerate(names) if gt in name), None)
        if rank is not None:
            reciprocal_ranks.append(1 / (rank + 1))

        # Calculate MAP (Mean Average Precision)
        relevant_at_positions = [1 if gt in name else 0 for name in names]
        
        if sum(relevant_at_positions) > 0:  # Check if there are any relevant documents
            average_precision = sum([relevant_at_positions[i] / (i + 1) for i in range(len(relevant_at_positions))]) / sum(relevant_at_positions)
            average_precisions.append(average_precision)
        else:
            average_precisions.append(0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0

    return map_score, mrr


if __name__ == "__main__":
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    map_score, mrr = compute_metrics(questions)

    print(f"\n📊 Hasil Evaluasi Retrieval:")
    print(f"MAP (Mean Average Precision): {map_score:.4f}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
