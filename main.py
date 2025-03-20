import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
import os

from search_agent import *

def init_session():
    st.session_state["initialized"] = True;
    st.session_state.messages = [];
    st.session_state.prompt = "";
    load_dotenv();
    key = os.getenv("GENIMI_API_KEY");
    genai.configure(api_key=key);


def fetch_llm_response(prompt, model="gemini-1.5-flash"):
    if "gemini" not in st.session_state:
        model = genai.GenerativeModel(model);
        chat = model.start_chat(history=list());
        st.session_state["gemini"] = {
            "model": model,
            "chat": chat
        }

    chat = st.session_state.gemini["chat"]
    response = chat.send_message(prompt)

    chat.history.append({"role": "user", "parts": prompt})
    chat.history.append({"role": "model", "parts": response.text})

    return response.text


if "initialized" not in st.session_state:
    init_session();


st.set_page_config(layout="wide")
st.title('W courses')

col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1], border=False)

with col1:
    platform = st.radio('Platform', ['Stepik', 'Coursera'])

with col2:
    course_name = st.text_input('Course name')

with col3:
    search_courses = st.button('Search courses')
    search_info = st.button('Search info')

with col4:
    use_rag = st.checkbox('Use rag')

with col5:
    language = st.radio('Language', ['English', 'Русский'])

if "answer" not in st.session_state:
    st.session_state["answer"] = list()

if search_courses and course_name and platform:
    response = None
    if use_rag:
        response = search_courses_rag(course_name, platform)
    else:
        response = search_courses_duck(course_name, platform)

    st.write(translate_text(response, language))
elif search_info and course_name and platform:
    st.write('nah')



