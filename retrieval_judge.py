import os
from typing import Dict, List
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from pydantic import BaseModel
import requests
import gzip
import json
from urllib.parse import urlparse
from datasets import load_dataset

from utils.evaluation import evaluate_retrieval
from utils.type import ResultFormat
from utils.preprocess import process_dataset

def is_url(string: str) -> bool:
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False


class RetrievalJudge:
    def __init__(self, source: str | None= None):
        self.data = ResultFormat(chunks={}, questions= []) # Default data format
        self.googleNQ = None

        if source is None:
            # Load from Hugging Face if no source is provided
            print("No source provided. Loading data from Hugging Face...")
            base_url = "https://huggingface.co/datasets/Quivr/Quivr_Google_NQ_dataset/resolve/main/evaluation_dataset.json?download=true"
            response = requests.get(base_url)
            data = response.json()
            self.data = ResultFormat(**data)
        else:
            # Check if source is a URL or a local file
            if is_url(source):
                print(f"Downloading data from URL: {source}")
                self._load_from_url(source)
            else:
                print(f"Loading data from local file: {source}")
                self._load_from_file(source)
        if isinstance(self.data, ResultFormat):
            print(f"Number of processed question loaded: {len(self.data.questions)}")
            print(f"Number of chunks loaded: {len(self.data.chunks)}")
        else:
            raise Exception("Invalid data format. Please provide a valid ResultFormat object stored link")


    def _load_from_url(self, url: str):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download data. Status code: {response.status_code}")
        data = response.json()
        self.data = ResultFormat(**data)
    
    def _load_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = ResultFormat(**data)

    
    def get_chunks(self) -> Dict:
        if isinstance(self.data, ResultFormat):
            return self.data.chunks
        else:
            raise Exception("Data is not in the expected ResultFormat")

    def set_chunks(self, chunks: List[List[str]]) -> ResultFormat:
        # Recompute the evaluation DS with ground truth ids and returns new test file
        if not self.googleNQ:
            self.googleNQ = self.get_googleNQ()
        data = process_dataset(self.googleNQ, chunks)
        self.data = ResultFormat(**data)
        return self.data

    def save_chunks(self, location):
        #save the chunks to a file
        pass

    def get_googleNQ(self):
        #if document dev-saùple.jsonl.gz does not exist, download it
        if not os.path.exists("dev-sample.jsonl.gz"):   
            print("Downloading dataset, this can take a minute ...")  
            url = "https://storage.googleapis.com/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz"
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

            # Save the downloaded content
            with open("dev-sample.jsonl.gz", "wb") as file:
                file.write(response.content)
            print("Dataset downloaded successfully.")

        # Unzip and read the content
        print("Unzipping NQ dataset, this can take a few seconds ...")  

        json_data = []
        with gzip.open("dev-sample.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                json_data.append(json.loads(line.strip()))
        
        print(f"Processed {len(json_data)} JSON lines.")
        self.googleNQ = json_data
        return self.googleNQ

    def get_NQ_dataset_pages(self):
        if not self.googleNQ:
            self.googleNQ = self.get_googleNQ()
        page_list = []
        for page in self.googleNQ:
            all_text: str = " ".join([el["token"] if not el["html_token"] else f'{el["token"]} \n' for el in page["document_tokens"]])
            page_list.append(all_text)
        return page_list

    def get_questions(self):
        if isinstance(self.data, ResultFormat):
            return [el.question for el in self.data.questions]

        else:
            raise Exception("Data is not in the expected ResultFormat")

    def evaluate(self, result: ResultFormat|Dict) -> Dict[str, float]:
        if isinstance(result, Dict):
            try:
                result = ResultFormat(**result)
            except Exception as e:
                raise Exception("Invalid result format. Please provide a valid ResultFormat object")

        assert isinstance(result, ResultFormat), "Invalid result format. Please provide a valid ResultFormat object"
        assert isinstance(self.data, ResultFormat), "Invalid result format. Please provide a valid ResultFormat object"

        return evaluate_retrieval(self.data.questions, result.questions)
