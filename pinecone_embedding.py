from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import embeddings
from database import get_dataset, init_pinecone_database

def dataset_embedding_and_upsert_to_pinecone(dataset_name, dataset_size, index_name, chunk_size):
    dataset = get_dataset(dataset_name, dataset_size)
    index = init_pinecone_database(index_name)
    
    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
    )
    
    # data 임베딩 실행
    process_data(dataset, text_splitter, embeddings, index)

# 배치 크기 설정
BATCH_SIZE = 100

# Pinecone에 데이터 업로드
def upload_to_pinecone(index, batch):
    index.upsert(vectors=batch)

# 임베딩 생성 함수 (멀티프로세싱용)
def create_embedding(chunk):
    return embeddings.embed_query(chunk)

# 메인 처리 함수
def process_data(dataset, text_splitter, index):
    batch = []
    total_processed = 0

    with tqdm(total=len(dataset) if hasattr(dataset, '__len__') else None) as pbar:
        for item in dataset:
            chunks = text_splitter.split_text(item['text'])

            # 멀티프로세싱을 사용한 임베딩 생성
            with ProcessPoolExecutor() as executor:
                chunk_embeddings = list(executor.map(create_embedding, chunks))

            for j, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                batch.append((f"{item['id']}_{j}", embedding, {
                    "text": chunk,
                    "title": item['title'],
                    "url": item['url']
                }))

                if len(batch) >= BATCH_SIZE:
                    upload_to_pinecone(index, batch)
                    batch = []

            total_processed += 1
            pbar.update(1)

            if total_processed % 1000 == 0:
                print(f"Processed {total_processed} items")

    # 남은 배치 처리
    if batch:
        upload_to_pinecone(index, batch)

    print("데이터 임베딩 및 저장 완료")