from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import embeddings, llm, answer_prompt
from database import get_dataset, init_pinecone_database
from factscore.factscorer import FactScorer

def question_and_answer(question_dataset_name, index_name, similarity_score_threshold, regenerate_question_max_attempts):
    index = init_pinecone_database(index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    if similarity_score_threshold:
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": similarity_score_threshold})
    else:
        retriever = vector_store.as_retriever()
    
    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
    factcheck_dataset_length = 3
    dataset = get_dataset(question_dataset_name, factcheck_dataset_length)

    for index, data in enumerate(dataset):
        print("======================")
        print(f"Test case : {index + 1} / {factcheck_dataset_length}")
        print("======================")
        result, documents = rag_with_fact_checking(data['question'], qa_chain, regenerate_question_max_attempts)
        if result:
            print(f"Final answer: {result}")
        else:
            print("Unable to find a factual answer.")

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

def rag_with_fact_checking(initial_question, qa_chain, max_attempts):
    current_question = initial_question

    for attempt in range(max_attempts):
        result = qa_chain.invoke({"input": current_question})
        answer = result['answer']
        
        documents = result['context']

        print(f"Attempt {attempt + 1}:")
        print(f"Question: {current_question}")
        print(f"Answer: {answer}")
        
        fs = FactScorer()    
        out = fs.get_score(topics=[current_question], generations=[answer], knowledge_source=documents, gamma=10)
        result_score = out["score"]
        print (f"Fact Score : {result_score}") # FActScore
        # print (out["init_score"]) # FActScore w/o length penalty
        # print (out["respond_ratio"]) # % of responding (not abstaining from answering)
        # print (out["num_facts_per_response"]) # average number of atomic facts per response

        if result_score > 0.9:
            print("Fact check passed. Returning answer.")            
            return {"question" : current_question, "answer" : answer}, documents
        else:
            print("Fact check failed. Generating new question.")
            current_question = generate_new_question(current_question)

    print(f"Max attempts ({max_attempts}) reached. No factual answer found.")
    return None, None