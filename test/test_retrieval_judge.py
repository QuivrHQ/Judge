import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
from retrieval_judge import RetrievalJudge
from utils.type import ResultFormat, QuestionFormat

@pytest.fixture
def sample_data():
    return {
        "chunks": {"1": "Sample chunk 1", "2": "Sample chunk 2"},
        "questions": [
            {"question": "Sample question 1", "chunk_ids": ["1"]},
            {"question": "Sample question 2", "chunk_ids": ["2"]}
        ]
    }

@pytest.fixture
def mock_requests_get(sample_data):
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_data
        mock_get.return_value = mock_response
        yield mock_get


def test_init_with_no_source(mock_requests_get, sample_data):
    judge = RetrievalJudge()
    
    # Assert that requests.get was called with the correct URL
    mock_requests_get.assert_called_once_with(
        "https://huggingface.co/datasets/Quivr/Quivr_Google_NQ_dataset/resolve/main/evaluation_dataset.json?download=true"
    )
    
    # Assert that judge.data is an instance of ResultFormat
    assert isinstance(judge.data, ResultFormat)
    
    # Assert the content of judge.data
    assert judge.data.chunks == sample_data["chunks"]
    assert len(judge.data.questions) == len(sample_data["questions"])

def test_init_with_url_source(mock_requests_get, sample_data):
    url = "http://example.com/data.json"
    judge = RetrievalJudge(url)
    assert isinstance(judge.data, ResultFormat)
    assert judge.data.chunks == sample_data["chunks"]
    assert len(judge.data.questions) == len(sample_data["questions"])
    mock_requests_get.assert_called_once_with(url)

def test_init_with_file_source(sample_data):
    file_path = "test_data.json"
    mock_file_content = json.dumps(sample_data)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('os.path.exists', return_value=True):
            judge = RetrievalJudge(file_path)
    
    assert isinstance(judge.data, ResultFormat)
    assert judge.data.chunks == sample_data["chunks"]
    assert len(judge.data.questions) == len(sample_data["questions"])

def test_get_chunks(sample_data):
    judge = RetrievalJudge()
    judge.data = ResultFormat(**sample_data)
    chunks = judge.get_chunks()
    assert chunks == sample_data["chunks"]

def test_get_questions(sample_data):
    judge = RetrievalJudge()
    judge.data = ResultFormat(**sample_data)
    questions = judge.get_questions()
    assert len(questions) == len(sample_data["questions"])

