from typing import List, Dict
from utils.type import QuestionFormat

def evaluate_retrieval(ground_truth: List[QuestionFormat], selected_chunk_ids: List[QuestionFormat]) -> Dict[str, float]:
    """
    Evaluate the performance of a retrieval system.

    Args:
    ground_truth: List[Dict]
        Each dict contains:
        - question: str
        - chunk_ids: List[str] of relevant chunk ids
    selected_chunk_ids: List[Dict]
        Each dict contains:
        - question: str
        - chunk_ids: List[str] of selected chunk ids

    Returns:
    dict: A dictionary containing the evaluation metrics
    """
    precisions = []
    recalls = []
    f1_scores = []
    average_precisions = []

    # Create dictionaries for easier lookup
    gt_dict = {item.question: set(item.chunk_ids) for item in ground_truth}
    selected_dict = {item.question: item.chunk_ids for item in selected_chunk_ids}

    for question, true_chunks in gt_dict.items():
        if question not in selected_dict:
            continue

        selected_chunks = selected_dict[question]

        # Calculate recall
        recall = len(true_chunks & set(selected_chunks)) / len(true_chunks) if true_chunks else 0
        recalls.append(recall)

    

        # Calculate average precision
        avg_precision = 0
        relevant_count = 0
        for i, chunk_id in enumerate(selected_chunks, 1):
            if chunk_id in true_chunks:
                relevant_count += 1
                avg_precision += relevant_count / i
        avg_precision /= len(true_chunks) if true_chunks else 1
        average_precisions.append(avg_precision)

    # Calculate mean metrics
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    mean_average_precision = sum(average_precisions) / len(average_precisions) if average_precisions else 0

    return {
        "mean_recall": mean_recall,
        "mean_average_precision": mean_average_precision
    }
