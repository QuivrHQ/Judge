from typing import Dict, List
from pydantic import BaseModel


class QuestionFormat(BaseModel):
    question: str
    chunk_ids: List[str]

class ResultFormat(BaseModel):
    chunks: Dict
    questions: List[QuestionFormat]