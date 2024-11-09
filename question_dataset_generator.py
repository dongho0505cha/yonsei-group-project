import random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import llm
from database import get_dataset

def create_questions_and_dataset_upload(dataset_name, dataset_size, chunk_size, repo_id):
    dataset = get_dataset(dataset_name, dataset_size)
    
    # 텍스트 분할기를 설정합니다.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    
    # 데이터셋 huggingface 업로드
    upload_dataset = Dataset.from_list(create_questions_dataset(dataset, text_splitter))
    upload_dataset.push_to_hub(repo_id=repo_id, token="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")

# 질문 생성 함수
def generate_question(text, question_type):
    prompt_template = PromptTemplate(
        input_variables=["text", "question_type"],
        template="Given the following text about a person: {text}\n\nGenerate a {question_type} question based on this information. Only provide the question itself, without any additional text or explanation."
    )
    prompt = prompt_template.format(text=text, question_type=question_type)

    if question_type == "simple":
      prompt += " Please create one simple fact-based question: the question should be one sentence and have a single, clear answer."
    elif question_type == "reasoning":
      prompt += " Please create one reasoning question that requires logical thinking and analysis. Questions require several steps in the thinking process, and answers require synthesizing information and drawing conclusions."
    elif question_type == "multi_context":
      prompt += " Please create one complex question. Questions explore the relationship between two topics and require respondents to consider multiple contexts."
    elif question_type == "ambiguous":
      prompt += " Please ask one intentionally vague question. Questions should be open to multiple interpretations or difficult to answer clearly without additional information."

    response = llm(prompt)

    # Check if response is an AIMessage and extract content if necessary
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = response

    # 응답에서 질문만 추출
    question = re.sub(r'^[^A-Za-z]+', '', response_text).strip()  # 시작 부분의 비알파벳 문자 제거
    question = re.sub(r'[^A-Za-z0-9\s\?\'\"\.\,\!\-].*', '', question)  # 끝 부분의 불필요한 문자 제거

    return question

def process_chunk(text):
    # 질문 유형
    question_types = [
        "simple",
        "reasoning",
        "multi_context",
        "ambiguous"
    ]
    
    question_type = random.choice(question_types)
    question = generate_question(text, question_type)
    return {
        "question_type": question_type,
        "question": question,
        "text": text,
    }

# 10개의 chunk를 처리하는 함수
def process_chunks(chunks):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        return [future.result() for future in as_completed(futures)]

def create_questions_dataset(dataset, text_splitter):
    # 질문 데이터셋 생성
    questions_dataset = []
    chunks_to_process = []

    for i, item in enumerate(dataset):
        if i >= 1000:  # 예시로 100개의 항목만 처리
            break

        chunks = text_splitter.split_text(item['text'])

        if chunks:  # chunks가 비어있지 않은 경우에만 처리
            # selected_chunk = random.choice(chunks)
            selected_chunk = chunks[0]
            chunks_to_process.append(selected_chunk)

        # 10개의 chunk가 모이면 병렬 처리
        if len(chunks_to_process) == 10 or (i == len(dataset) - 1 and chunks_to_process):
            questions_dataset.extend(process_chunks(chunks_to_process))
            chunks_to_process = []

    # 마지막으로 남은 chunks 처리 (10개 미만일 경우)
    if chunks_to_process:
        questions_dataset.extend(process_chunks(chunks_to_process))
        
    return questions_dataset