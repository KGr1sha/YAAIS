import os
from dotenv import load_dotenv
import asyncio
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import torch


platforms = {
    "Stepik": "https://stepik.org/catalog",
    "Coursera": "https://www.coursera.org/courses"
}

search_prompt = """
You are an AI assistant helping users find online courses. The user wants to find a course based on the following criteria:

- Topic: {topic}
- Platform: {platform}
- Cost: between {cost[0]} and {cost[1]} dollars
- Difficulty: {difficulty} (Beginner, Intermediate, Advanced, or Any)
- Rating: at least {rating}
- Duration: between {duration[0]} and {duration[1]} hours

Please search the knowledge base and return a list of all courses that match these criteria.

Requirements:
- Include course titles and short descriptions.
- Mention the course author or organization, if available.
- Provide direct links to the courses, if possible.
"""

load_dotenv()

def create_rag_agent(platform_name: str):
    vector_store_path = "vector_store" + platform_name
    vector_store = None
    login(os.getenv('HUGGING_FACE_TOKEN'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device}
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", model_kwargs=model_kwargs)

    if os.path.exists(vector_store_path):
        print(f"Loading existing vector store from {vector_store_path}")

        vector_store = FAISS.load_local(
            vector_store_path,
            emb_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("Vector store not found, creating a new one.")
        loader = RecursiveUrlLoader(platforms[platform_name])
        docs = loader.load()
        text_contents = [doc.page_content for doc in docs]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        split_documents = splitter.create_documents(text_contents)
        print(len(split_documents))
        vector_store = FAISS.from_documents(
            split_documents[:100],
            emb_model
        )
        vector_store.save_local(vector_store_path)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        k=3,
        score_threshold=None
    )

    llm = ChatGroq(model="llama3-70b-8192")
    tool = create_retriever_tool(
        retriever,
        platform_name+"retriever",
        f"searches courses on a platform called {platform_name}"
    )
    return create_react_agent(llm, [tool])


def search_courses_rag(agent, criteria: dict) -> str:
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": search_prompt.format(
                topic=criteria['topic'],
                platform=criteria['platform'],
                cost=criteria['cost'],
                difficulty=criteria['difficulty'],
                rating=criteria['rating'],
                duration=criteria['duration']
            )
        }]
    })

    return result["messages"][-1].content


def create_search_agent():
    load_dotenv()
    llm = ChatGroq(model="llama3-70b-8192")
    search = DuckDuckGoSearchRun()
    return create_react_agent(llm, [search])


def search_courses(agent, criteria: dict) -> str:
    response = agent.invoke(
        {"messages": [{
            "role": "user",
            "content": search_prompt.format(
                topic=criteria['topic'],
                platform=criteria['platform'],
                cost=criteria['cost'],
                difficulty=criteria['difficulty'],
                rating=criteria['rating'],
                duration=criteria['duration']
            )
        }]},
        stream_mode="values"
    )
    return response['messages'][-1].content


def translate_text(text: str, target_language: str) -> str:
    llm_for_translate = ChatGroq(model="llama3-70b-8192")
    prompt = f"Translate following text to {target_language}: \"{text}\""
    response = llm_for_translate.invoke(prompt)
    return response.content


def get_course_info(agent, course_name: str, platform: str) -> str:
    query = f"Найди информацию о курсе '{course_name}' на платформе {platform}. Верни ссылку на курс, описание и ключевые темы."
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content

