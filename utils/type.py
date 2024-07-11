from typing import Dict, List
from pydantic import BaseModel

class ResultType(BaseModel):
    all_recall: List[Dict[str, float]]
    mean_recall: Dict[str, float]
    mean_map_metric: float

class ReferenceType(BaseModel):
    _id: str
    text:str
    question:str
    long_answers:list[str]
    short_answers:list[str]