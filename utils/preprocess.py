from typing import List, Dict

def get_candidate_text(candidate: Dict, document_tokens: List[Dict]) -> str:
    tokens = document_tokens[candidate['start_token']:candidate['end_token']]
    return ' '.join(token['token'] for token in tokens if not token['html_token']) + '\n'

def process_dataset(dataset: List[Dict]) -> tuple[Dict[str, str], List[Dict]]:
    chunk_dict = {}
    questions = []
    
    for i, element in enumerate(dataset):
        start_index = len(chunk_dict)
        
        # Extract chunks
        new_chunks = {
            f"{i}.{j}": get_candidate_text(candidate, element['document_tokens'])
            for j, candidate in enumerate(element['long_answer_candidates'])
            if candidate['top_level']
        }
        chunk_dict.update(new_chunks)
        
        # Process annotations
        chunk_ids = {
            f"{i}.{annotation['long_answer']['candidate_index']}"
            for annotation in element['annotations']
            if annotation['long_answer']['candidate_index'] != -1
        }
        
        questions.append({
            "question": element['question_text'],
            "chunk_ids": list(chunk_ids)
        })
    
    return chunk_dict, questions


if __name__ == "__main__":
    import json
    file_path = 'dev-sample.jsonl'
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    chunk_dict, questions = process_dataset(dataset)

    #Save preprocessed dataset
    json.dump(chunk_dict, open('processed_data/text_corpus.json', 'w'),  indent=2)
    json.dump(questions, open('processed_data/questions_corpus.json', 'w'), indent=2)