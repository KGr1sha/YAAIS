import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.prebuilt import create_react_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.vectorstores import FAISS

def search_courses(course_name, platform):
    load_dotenv()
    llm = ChatGroq(model="llama3-70b-8192")
    python_tool = PythonREPLTool()

    login(os.getenv('HUGGING_FACE_TOKEN'))

    emb_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    search = DuckDuckGoSearchRun()
    agent_with_extractor = create_react_agent(
        llm,
        [search]
    )
    query_2 = f"Найди информацию о курсе '{course_name}' на платформе {platform}. Верни ссылку на курс, описание и ключевые темы."

    result_2 = agent_with_extractor.invoke({
        "messages": [{"role": "user", "content": query_2}]
    })

    response_text_2 = result_2["messages"][-1].content

    return response_text_2


def get_course_info(course_name, platform):
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
    return 'not implemented'

if __name__ == '__main__':
    print(search_courses('python programmer', 'stepik'))
