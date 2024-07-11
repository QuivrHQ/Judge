from typing import List, Dict
from utils.type import ReferenceType, ResultType
import difflib
import tqdm
def find_longest_common_substring(answer, chunk):
    # Create a SequenceMatcher object
    matcher = difflib.SequenceMatcher(None, chunk, answer)
    
    # Find the longest matching block
    match = matcher.find_longest_match(0, len(chunk), 0, len(answer))
    
    return match.size / len(answer)

def compute_map_like_metric(results):
    precision_at_k = {}
    total_precision = 0
    
    # Calculate precision at each k
    for k in range(1, len(results) + 1):
        recall_at_k = results[f'top_{k}']
        precision_at_k[k] = recall_at_k / k
        total_precision += precision_at_k[k]

    # Calculate Average Precision (AP)
    ap = total_precision / len(results)

    return ap

def evaluate_one_retrieval(response: list[str], ground_truth: ReferenceType): #FIXME precise typing
    results = {}
    text = ""

    for i, chunk in enumerate(response):
        text += chunk + " "
        if ground_truth.short_answers:
            results_short = []

            for short_answer in ground_truth.short_answers:
                match_len = find_longest_common_substring(short_answer.strip(), text)
                results_short.append(match_len)

            results[f"top_{i+1}"] = max(results_short)

        elif len(ground_truth.long_answers) > 0:
            results_long = []
            for long_answer in ground_truth.long_answers:
                match_len = find_longest_common_substring(long_answer, text)
                results_long.append(match_len)
            results[f"top_{i+1}"] = sum(results_long) / len(results_long)
    
    return results

def evaluate_all_retrieval(responses: List[List[str]], ground_truths: List[ReferenceType]) -> ResultType:# -> tuple[None, None, None] | dict[str, Any]:
    all_results = []
    map_metrics = []

    for response,elements in tqdm.tqdm(zip(responses, ground_truths),  total=len(responses)):
        results = evaluate_one_retrieval(response, elements)
        if results:
            all_results.append(results)
            map_metric = compute_map_like_metric(results)
            map_metrics.append(map_metric)
    
    # Compute mean recall per k
    if not all_results:
        return ResultType(all_recall = [], mean_recall = {}, mean_map_metric= 0)

    sum_recall = {}
    for results in all_results:
        for k, v in results.items():
            if k not in sum_recall:
                sum_recall[k] = 0
            sum_recall[k] += v


    mean_recall = {k: v / len(all_results) for k, v in sum_recall.items()}

    # Compute mean average precision (MAP-like metric)
    mean_map_metric = sum(map_metrics) / len(map_metrics)

    return ResultType(**{"all_recall": all_results, "mean_recall": mean_recall, "mean_map_metric" : mean_map_metric})
