import time
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

def get_dataset(dataset_name, dataset_size):
    # 위키피디아 인물 데이터셋 로드
    dataset = load_dataset(dataset_name, split="train")

    # dataset 1000개만 저장
    dataset = dataset.select(range(dataset_size))
    
    return dataset

def init_pinecone_database(index_name):
    # retriever 설정
    pc = Pinecone(api_key="bf237795-da71-4661-9497-55e96c069b6f")
    # index_name = "wikipedia-persons"
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    
    return index