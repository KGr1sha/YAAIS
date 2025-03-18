import streamlit as st
import asyncio
import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv
import os

from search_agent import search_courses

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



st.title('Search ur shit')

col1, col2, col3 = st.columns([1, 1, 1], border=True)

with col1:
    platform = st.radio('Platform', ['Stepik', 'Coursera'])

with col2:
    course_name = st.text_input('Course name')

with col3:
    search_courses = st.button('Search courses')
    search_info = st.button('Search info')


if search_courses and course_name and platform:
    st.write(search_courses(course_name, platform))
elif search_info and course_name and platform:
    st.write('nah')

