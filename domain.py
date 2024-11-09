from pydantic import BaseModel
from typing import Optional

class EmbeddingRequest(BaseModel):
    dataset_name: str
    dataset_size: Optional[int] = 1000
    chunk_size: Optional[int] = 2000
    index_name: str    
    
class GenerateQuestionsRequest(BaseModel):
    dataset_name: str
    dataset_size: Optional[int] = 1000
    chunk_size: Optional[int] = 10000
    upload_repo_id: str
    
class RagExecuteRequest(BaseModel):
    question_dataset_name: str
    index_name: str
    similarity_score_threshold: Optional[float] = None
    regenerate_question_max_attempts: Optional[int] = 3
    
class Question(BaseModel):
    question_type: str
    question: str
    text: str