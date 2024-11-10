from fastapi import FastAPI
from domain import EmbeddingRequest, GenerateQuestionsRequest, RagExecuteRequest
from pinecone_embedding import dataset_embedding_and_upsert_to_pinecone
from question_dataset_generator import create_questions_and_dataset_upload
from rag_test_executor import question_and_answer
from factscore.atomic_facts import AtomicFactGenerator
from factscore.factscorer import FactScorer

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
    result, documents = question_and_answer(request.question_dataset_name, request.index_name, request.similarity_score_threshold, request.regenerate_question_max_attempts)

    generation = """Albert Einstein made several groundbreaking contributions to physics, most notably through his theory of relativity. In 1905, he published three epochal papers in the journal Annalen der Physik, which included his special theory of relativity. This theory fundamentally changed the understanding of space and time and was soon widely accepted, particularly in Germany, thanks in part to the influence of Max Planck.

Einstein also introduced the concept of light quanta, or photons, which was initially met with skepticism, including from Planck himself. His work on the photoelectric effect, which demonstrated the particle-like properties of light, later earned him 
the Nobel Prize in Physics in 1921.

In 1910, Einstein identified the anomalous behavior of specific heat at low temperatures, which classical physics could not explain. This led to his collaboration with Planck and others at the First Solvay Conference in 1911, where he was able to convince Planck of the validity of his ideas.

Einstein's contributions extended beyond special relativity; he later developed the general theory of relativity, which provided a new understanding of gravitation. His work laid the foundation for modern physics and has had a lasting impact on various fields, including cosmology and quantum mechanics."""

    # generator = AtomicFactGenerator()
    # atomic_facts, para_breaks = generator.run(generation)
    
    # print(atomic_facts)
    # print(para_breaks)
    
    fs = FactScorer()    
    out = fs.get_score(topics=[result['question']], generations=[result['answer']], knowledge_source=documents, gamma=10)
    print (out["score"]) # FActScore
    print (out["init_score"]) # FActScore w/o length penalty
    print (out["respond_ratio"]) # % of responding (not abstaining from answering)
    print (out["num_facts_per_response"]) # average number of atomic facts per response
    
    return {"Result": "Success"}