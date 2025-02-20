from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.retrieval import create_retrieval_multi_query_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import embeddings, llm, answer_prompt
from database import get_dataset, init_pinecone_database
from factscore.factscorer import FactScorer
import csv
import time
import concurrent.futures
import os


def process_data(index, data, qa_chain, regenerate_question_max_attempts):
    print("======================")
    print(f"Test case : {index + 1}")
    print("======================")

    # rag_with_fact_checking 함수 호출
    result, documents, attempts = rag_with_fact_checking(data['question'], qa_chain, regenerate_question_max_attempts, index + 1)

    if result:
        print(f"Final answer: {result}")
    else:
        print("Unable to find a factual answer.")
    
    # 각 시도 데이터를 반환
    return index, attempts

def question_and_answer(question_dataset_name, index_name, similarity_score_threshold, regenerate_question_max_attempts):
    index = init_pinecone_database(index_name)
    print(f"{os.cpu_count() * 5}")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    if similarity_score_threshold:
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": similarity_score_threshold})
    else:
        retriever = vector_store.as_retriever()
    
    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    multi_qa_chain = create_retrieval_multi_query_chain(retriever, combine_docs_chain)


    factcheck_dataset_length = 100
    dataset = get_dataset(question_dataset_name, factcheck_dataset_length)

    outputFileName = f"output{round(time.time(), 1)}.csv"
    with open(outputFileName, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question_id", "question", "attempt", "answer", "fact_score", "retrieval_time", "factchecking_time", "atomic_token", "checking_token"])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers= (os.cpu_count() * 5)/2) as executor:
            # process_data 함수를 병렬로 호출
            futures = [
                executor.submit(process_data, index, data, qa_chain, regenerate_question_max_attempts)
                for index, data in enumerate(dataset)
            ]

            # multi query request..
            # futures = [
            #     executor.submit(process_data, index, data, multi_qa_chain, regenerate_question_max_attempts)
            #     for index, data in enumerate(dataset)
            # ]

            # 완료된 작업에서 결과를 수집하여 CSV 파일에 기록
            for future in concurrent.futures.as_completed(futures):
                index, attempts = future.result()
                for attempt_data in attempts:
                    writer.writerow([
                        index + 1,
                        attempt_data["question"],
                        attempt_data["attempt"],
                        attempt_data["answer"],
                        attempt_data["fact_score"],
                        attempt_data["retrieval_time"],
                        attempt_data["factchecking_time"],
                        attempt_data["atomic_token"],
                        attempt_data["factchecking_token"]
                    ])


        # for index, data in enumerate(dataset):
        #     print("======================")
        #     print(f"Test case : {index + 1} / {factcheck_dataset_length}")
        #     print("======================")


        #     result, documents, attempts = rag_with_fact_checking(data['question'], qa_chain, regenerate_question_max_attempts, index+1)
           
        #     for attempt_data in attempts:
        #         writer.writerow([index +1, attempt_data["question"], attempt_data["attempt"], attempt_data["answer"], attempt_data["fact_score"], attempt_data["retrieval_time"], attempt_data["factchecking_time"], attempt_data["atomic_token"], attempt_data["factchecking_token"]])

def fact_check(question, answer, context):
    fact_check_prompt = PromptTemplate(
        input_variables=["question", "answer", "context"],
        template="Question: {question}\nAnswer: {answer}\nContext: {context}\n\nIs the answer factually correct and supported by the context? Respond with 'Yes' or 'No' and explain why."
    )

    # fact_checker = OpenAI(temperature=0)
    fact_check_chain = LLMChain(llm=llm, prompt=fact_check_prompt)
    # result = fact_checker(fact_check_prompt.format(question=question, answer=answer, context=context))
    result = fact_check_chain.run(question=question, answer=answer, context=context)

    return result.strip().lower().startswith("yes")

def generate_new_question(original_question):
    new_question_prompt = PromptTemplate(
        input_variables=["original_question"],
        template="The following question did not yield a factual answer: '{original_question}'. Please generate a more specific or refined version of this question."
    )

    question_generator = LLMChain(llm=llm, prompt=new_question_prompt)
    new_question = question_generator.run(original_question=original_question)

    return new_question.strip()

def rag_with_fact_checking(initial_question, qa_chain, max_attempts, index):
    current_question = initial_question
    attempt_data = []
    for attempt in range(max_attempts):
        retrieval_start_time = time.time()
        result = qa_chain.invoke({"input": current_question})
        answer = result['answer']
        retrieval_end_time = time.time()
        documents = result['context']

        print(f"Attempt {attempt + 1}:")
        print(f"Question: {current_question}")
        print(f"Answer: {answer}")
        factchecking_start_time = time.time()
        fs = FactScorer()    
        out = fs.get_score(topics=[current_question], generations=[answer], knowledge_source=documents, gamma=10)
        result_score = out["score"]
        print (f"Fact Score : {result_score}") # FActScore
        # print (out["init_score"]) # FActScore w/o length penalty
        # print (out["respond_ratio"]) # % of responding (not abstaining from answering)
        # print (out["num_facts_per_response"]) # average number of atomic facts per response
        factchecking_end_time = time.time()
        attempt_data.append({
            "question_id": index,
            "question": current_question,
            "attempt": attempt + 1,
            "answer": answer,
            "fact_score": result_score,
            "retrieval_time": round(retrieval_end_time - retrieval_start_time, 2),
            "factchecking_time": round(factchecking_end_time - factchecking_start_time, 2),
            "atomic_token":out["atomictoken"],
            "factchecking_token": out["factcheckingtoken"]
        })

        if result_score > 0.9:
            print("Fact check passed. Returning answer.")            
            return {"question" : current_question, "answer" : answer}, documents, attempt_data
        else:
            print("Fact check failed. Generating new question.")
            current_question = generate_new_question(current_question)

    print(f"Max attempts ({max_attempts}) reached. No factual answer found.")
    return None, None, attempt_data