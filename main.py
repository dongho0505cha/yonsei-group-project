from fastapi import FastAPI
from domain import EmbeddingRequest, GenerateQuestionsRequest, RagExecuteRequest
from pinecone_embedding import dataset_embedding_and_upsert_to_pinecone
from question_dataset_generator import create_questions_and_dataset_upload
from rag_test_executor import question_and_answer

app = FastAPI()

@app.post("/embedding", tags=["Dataset Embedding"])
def data_embedding(request: EmbeddingRequest):
    dataset_embedding_and_upsert_to_pinecone(request.dataset_name, request.dataset_size, request.chunk_size, request.index_name)
    return {"Result": "Success"}

@app.post("/generate_questions", tags=["Questions Generation"])
def generate_questions(request: GenerateQuestionsRequest):
    create_questions_and_dataset_upload(request.dataset_name, request.dataset_size, request.chunk_size, request.upload_repo_id)
    return {"Result": "Success"}

@app.post("/rag_execute", tags=["RAG execute"])
def rag_execute(request: RagExecuteRequest):
    question_and_answer(request.question_dataset_name, request.index_name, request.similarity_score_threshold, request.regenerate_question_max_attempts)
    return {"Result": "Success"}