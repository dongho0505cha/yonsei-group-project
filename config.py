from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

# Azure OpenAIEmbeddings 설정
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://jp-britymeeting-dev-1.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15",
    api_key="ef4e64c7a44b4dc9b52bf2939ca52480",
    api_version="2023-05-15",
    deployment="text-embedding-3-small",
    model="text-embedding-3-small",
    # dimensions=512
)

# Azure ChatOpenAI 설정
llm = AzureChatOpenAI(
    azure_endpoint="https://jp-britymeeting-dev-1.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview",
    api_key="ef4e64c7a44b4dc9b52bf2939ca52480",
    api_version="2024-08-01-preview",
    deployment_name="gpt-4o-mini",
    model_name="gpt-4o-mini",
    temperature=0
)

# 프롬프트를 생성합니다.
answer_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

#Context:
{context}

#Question:
{input}

#Answer:"""
)