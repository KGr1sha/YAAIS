import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain_community.vectorstores import FAISS

def search_courses_rag(course_name: str, platform_url: str) -> str:
    load_dotenv()

    loader = RecursiveUrlLoader(platform_url)
    docs = loader.load()
    llm = ChatGroq(model="llama3-70b-8192")

    text_contents = [doc.page_content for doc in docs]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    split_documents = splitter.create_documents(text_contents)

    login(os.getenv('HUGGING_FACE_TOKEN'))
    emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    vector_store = FAISS.from_documents(split_documents, emb_model)
    vector_store.save_local("vector_store")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        k=3,
        score_threshold=None,
    )
    tool = create_retriever_tool(
        retriever,
        "https://stepik.org/catalog",
        "Найди и верни курсы на stepik",
    )
    agent_chain = create_react_agent(llm, [tool])

    query = f"Подбери мне онлайн-курсы по {course_name} с практическими заданиями."
    result = agent_chain.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    return result["messages"][-1].content


def search_courses_duck(course_name, platform):
    load_dotenv()
    llm = ChatGroq(model="llama3-70b-8192")
    search = DuckDuckGoSearchRun()
    agent_with_extractor = create_react_agent(llm, [search])
    query_2 = f"Найди курсы по {course_name} на платформе {platform}. К каждому из курсов прикрепи ссылку на сам курс."
    response = agent_with_extractor.invoke(
        {"messages": [{"role": "user", "content": query_2}]},
        stream_mode="values"
    )
    return response['messages'][-1].content
    

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
