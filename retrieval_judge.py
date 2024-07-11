import os
from typing import List
import requests
import gzip
import json
from urllib.parse import urlparse
from datasets import load_dataset

from utils.evaluation import evaluate_all_retrieval
from utils.type import ReferenceType
from utils.plotting import plot_mean_recall

def is_url(string: str) -> bool:
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False


class RetrievalJudge:
    def __init__(self, ref_source : str = "https://huggingface.co/datasets/Quivr/Quivr_Google_NQ_dataset/resolve/main/evaluation_dataset.jsonl?download=true"):
        self.results = None
        if is_url(ref_source):
            print("Downloading reference data ...")
            #curl ref
            response = requests.get(ref_source)
            #read jsonl
            self.ref_data = []
            for line in response.iter_lines():
                article = json.loads(line)
                self.ref_data.append(ReferenceType(**article))
        else:
            with open(ref_source, "r") as f:
                self.ref_data = []
                for line in f:
                    article = json.loads(line)
                    self.ref_data.append(ReferenceType(**article))
    
    def get_pages(self):
        return [page.text for page in self.ref_data]

    def evaluate(self, results: List[List[str]], visualize = False):
        self.results = evaluate_all_retrieval(results, self.ref_data)
        if visualize:
            plot_mean_recall([self.results.mean_recall])
        return self.results


    
