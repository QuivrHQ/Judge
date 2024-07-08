from typing import List, Dict

def get_candidate_text(candidate: Dict, document_tokens: List[Dict]) -> str:
    tokens = document_tokens[candidate['start_token']:candidate['end_token']]
    return ' '.join(token['token'] for token in tokens if not token['html_token']) + '\n'

def process_dataset(dataset: List[Dict]) -> Dict[str, Dict]:
    processed_data = {
        "chunks": {},
        "questions": []
    }
    
    for i, element in enumerate(dataset):        
        # Extract chunks
        new_chunks = {
            f"{i}.{j}": get_candidate_text(candidate, element['document_tokens'])
            for j, candidate in enumerate(element['long_answer_candidates'])
            if candidate['top_level']
        }
        processed_data["chunks"].update(new_chunks)
        
        # Process annotations
        chunk_ids = {
            f"{i}.{annotation['long_answer']['candidate_index']}"
            for annotation in element['annotations']
            if annotation['long_answer']['candidate_index'] != -1
        }
        
        processed_data["questions"].append({
            "question": element['question_text'],
            "chunk_ids": list(chunk_ids)
        })
    
    return processed_data


if __name__ == "__main__":
    import json
    file_path = 'dev-sample.jsonl'
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    processed_data = process_dataset(dataset)

    #Save preprocessed dataset
    json.dump(processed_data, open('processed_data/evaluation_dataset.json', 'w'),  indent=2)
