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

search_prompt = """You are an AI assistant that helps users find online courses.  
The user wants to find a course on a specific platform: {platform_name}.  
Your task is to search the knowledge base for relevant courses and provide a concise, structured response.  

Requirements:  
- List the most relevant courses.  
- Include course titles and short descriptions.  
- If possible, mention the course author or organization.  
- Provide direct links if available.  

User query: {query}
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
    agent_chain = create_react_agent(llm, [tool])
    return agent_chain


stepik_agent = create_rag_agent("Stepik")
coursera_agent = create_rag_agent("Coursera")
    

# voodoo
#try:
#    asyncio.get_running_loop()
#except RuntimeError:
#    asyncio.set_event_loop(asyncio.new_event_loop())

def search_courses_rag(course_name: str, platform_name: str) -> str:
    query = f"Подбери мне онлайн-курсы по {course_name} с практическими заданиями."
    result = 'wtf is this platfurmf dude'
    if platform_name == "Stepik":
        result = stepik_agent.invoke({
            "messages": [{
                "role": "user",
                "content": search_prompt.format(platform_name="Stepik", query=query)
            }]
        })
    elif platform_name == "Coursera":
        result = coursera_agent.invoke({
            "messages": [{
                "role": "user",
                "content": search_prompt.format(platform_name="Coursera", query=query)
            }]
        })
    else:
        pass

    return result["messages"][-1].content


def search_courses_duck(course_name, platform):
    load_dotenv()
    llm = ChatGroq(model="llama3-70b-8192")
    search = DuckDuckGoSearchRun()
    agent_with_extractor = create_react_agent(llm, [search])
    query = f"Найди курсы по {course_name} на платформе {platform}. К каждому из курсов прикрепи ссылку на сам курс."
    response = agent_with_extractor.invoke(
        {"messages": [{
            "role": "user",
            "content": search_prompt.format(platform_name=platform, query=query)
        }]},
        stream_mode="values"
    )
    return response['messages'][-1].content
    


def translate_text(text: str, target_language: str) -> str:
    llm_for_translate = ChatGroq(model="llama3-70b-8192")

    prompt = f"Translate following text to {target_language}: \"{text}\" Translation:"

    response = llm_for_translate.invoke(prompt)
    print(type(response))
    print(response)
    return response.content
    #    if isinstance(response, dict) and "content" in response:
#        translated_text = response["content"].strip()
#    else:
#        translated_text = str(response).strip()
#
#    return translated_text


def get_course_info(course_name: str, platform: str) -> str:
    load_dotenv()
    llm = ChatGroq(model="llama3-70b-8192")
    search = DuckDuckGoSearchRun()
    agent_with_extractor = create_react_agent(
        llm,
        [search]
    )
    query = f"Найди информацию о курсе '{course_name}' на платформе {platform}. Верни ссылку на курс, описание и ключевые темы."
    response = agent_with_extractor.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return response["messages"][-1].content


def get_course_plan(course_name: str, platform: str) -> str:
    pass

#    try:
#
#        query_3 = f"Проанализируй следующую информацию о курсе '{course_name}' на Stepik:\n{response_text_2}\n\nПредложи учебный план, учитывая следующие параметры: продолжительность курса, сложность, ключевые темы и практические задания. Предложи расписание занятий и советы для эффективного обучения."
#        result_3 = agent_with_extractor.invoke({
#            "messages": [{"role": "user", "content": query_3}]
#        })
#
#        response_text_3 = result_3["messages"][-1].content
#
#        print("Результат третьего запроса (учебный план):")
#        print(response_text_3)
#
#    except ValueError as e:
#        print(f"Ошибка при обработке данных: {e}")
#    except Exception as e:
#        print(f"Произошла непредвиденная ошибка: {e}")
#    return result["messages"][-1].content




if __name__ == '__main__':
    print(search_courses_duck('python programmer', 'stepik'))
