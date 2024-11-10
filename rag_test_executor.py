from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import embeddings, llm, answer_prompt
from database import init_pinecone_database

def question_and_answer(question_dataset_name, index_name, similarity_score_threshold, regenerate_question_max_attempts):
    index = init_pinecone_database(index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    if similarity_score_threshold:
        retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": similarity_score_threshold})
    else:
        retriever = vector_store.as_retriever()
    
    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # TODO : question_dataset 가져오기
    
    # 1 pass case question
    initial_question = "Tell me about Einstein’s achievements"
    # 1 fail, 1 pass case question
    # initial_question = "Describe the area where Einstein lived."
    # 5 fail case question
    # initial_question = "How much is Einstein's bounty?"

    result = rag_with_fact_checking(initial_question, qa_chain, regenerate_question_max_attempts)
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
        context = "\n".join([document.page_content for document in result['context']]) if result['context'] else ""

        print(f"Attempt {attempt + 1}:")
        print(f"Question: {current_question}")
        print(f"Answer: {answer}")

        if fact_check(current_question, answer, context):
            print("Fact check passed. Returning answer.")
            return answer
        else:
            print("Fact check failed. Generating new question.")
            current_question = generate_new_question(current_question)

    print(f"Max attempts ({max_attempts}) reached. No factual answer found.")
    return None