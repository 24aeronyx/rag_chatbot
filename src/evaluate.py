import numpy as np

def evaluate(query, ground_truth, predicted):
    """
    Evaluate chatbot using mAP and MRR metrics.
    Parameters:
    - query: Input query string
    - ground_truth: List of expected responses (list of lists, each containing relevant documents)
    - predicted: List of predicted responses from chatbot (list of lists, each containing predicted documents)
    """
    # Mean Average Precision (mAP) calculation
    ap_scores = []
    for gt, pred in zip(ground_truth, predicted):
        # Calculate average precision for this query
        relevant_docs = [1 if doc in gt else 0 for doc in pred]  # Mark relevant docs as 1, others as 0
        precision_at_k = np.cumsum(relevant_docs) / (np.arange(len(pred)) + 1)  # Precision at each rank
        ap = np.mean(precision_at_k * relevant_docs)  # Average Precision
        ap_scores.append(ap)

    map_score = np.mean(ap_scores)  # Mean Average Precision

    # Mean Reciprocal Rank (MRR) calculation
    mrr_scores = []
    for gt, pred in zip(ground_truth, predicted):
        rank = next((i for i, doc in enumerate(pred) if doc in gt), None)
        if rank is not None:
            mrr_scores.append(1 / (rank + 1))  # Reciprocal Rank (1-based index)
        else:
            mrr_scores.append(0)

    mrr_score = np.mean(mrr_scores)  # Mean Reciprocal Rank

    return map_score, mrr_score


if __name__ == "__main__":
    # Contoh evaluasi
    query = "Apa itu abetalipoproteinemia?"
    ground_truth = [["Abetalipoproteinemia adalah kelainan bawaan langka..."]]
    predicted = [["Abetalipoproteinemia adalah kelainan bawaan langka..."]]
    
    map_score, mrr_score = evaluate(query, ground_truth, predicted)
    print(f"mAP: {map_score}, MRR: {mrr_score}")
