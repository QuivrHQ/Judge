from typing import List, Dict
import re
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_candidate_text(candidate: Dict, document_tokens: List[Dict]) -> str:
    tokens = document_tokens[candidate['start_token']:candidate['end_token']]
    return ' '.join(token['token'] for token in tokens if not token['html_token']) + '\n'

def process_dataset_simple(dataset: List[Dict]) -> Dict[str, Dict]:
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

def extract_one(data: Dict, texts: List[Document], article_id: int) -> Dict:
    idx = 0
    chunks: Dict[str, str] = {}
    long_answers = [el["long_answer"] for el in data["annotations"]]
    short_answers = [e for el in data["annotations"] if el["short_answers"] for e in el["short_answers"]]
    answer_range = set()
    response_chunk_ids = set()

    if short_answers:
        for ans in short_answers:
            answer_range.update(range(ans["start_token"], ans["end_token"]))
    else:
        for ans in long_answers:
            if ans['candidate_index'] != -1:
                answer_range.update(range(ans["start_token"], ans["end_token"]))

    for text in texts:
        chunk_id = f"{article_id}.{len(chunks)}"
        content = text.page_content
        new_content = []

        tokens = re.split(r'(\s+|(?:<[^>]*>))', content)
        tokens = [token for token in tokens if token.strip()]

        for token in tokens:
            if idx in answer_range:
                response_chunk_ids.add(chunk_id)
            if token == "\n":
                pass
            #print("one:",data["document_tokens"][idx]["token"],";two", token, " so ", data["document_tokens"][idx]["token"] == token)
            if idx < len(data["document_tokens"]) and data["document_tokens"][idx]["token"] == token:
                if not data["document_tokens"][idx]["html_token"]:
                    new_content.append(token)
                idx += 1

        if new_content:
            chunks[chunk_id] = " ".join(new_content)

    return {
        "question": data["question_text"],
        "response_chunk_ids": list(response_chunk_ids),
        "chunks": chunks
    }

def process_dataset(dataset: List[Dict], text_list: List[List[Document]]) -> Dict:
    processed_data = {
        "chunks": {},
        "questions": []
    }
    for i, (data, texts) in enumerate(zip(dataset, text_list)):
        result = extract_one(data, texts, i)
        processed_data["questions"].append({
            "question": result["question"],
            "chunk_ids": result["response_chunk_ids"]
        })
        processed_data["chunks"].update(result["chunks"])
    
    return processed_data



if __name__ == "__main__":
    import json
    file_path = 'dev-sample.jsonl'
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            dataset.append(data)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    )
    text_list = []
    for page in dataset:
        all_text: str = " ".join([el["token"] if not el["html_token"] else f'{el["token"]} \n' for el in page["document_tokens"]])
        texts = text_splitter.create_documents([all_text])
        text_list.append(texts)

    processed_data = process_dataset(dataset, text_list)

    print(f"Number of processed question: {len(processed_data['questions'])}")
    print(f"Number of chunks: {len(processed_data['chunks'])}")

    #Save preprocessed dataset
    json.dump(processed_data, open('processed_data/evaluation_dataset.json', 'w'),  indent=2)
