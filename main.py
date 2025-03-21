import os
import nest_asyncio
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

from search_agent import *

def init_session():
    load_dotenv();
    st.session_state["initialized"] = True;

    st.session_state.rag_stepik_agent = create_rag_agent('Stepik')
    st.session_state.rag_coursera_agent = create_rag_agent('Coursera')
    st.session_state.search_agent = create_search_agent()


if "initialized" not in st.session_state:
    init_session();


st.set_page_config(layout="wide")
st.title('W courses')

col1, col2, col3 = st.columns([2, 3, 1], border=False)

with col1:
    platform = st.radio('Platform', ['Stepik', 'Coursera', 'Other'])
    if platform == 'Other':
        platform = st.text_input('platform', label_visibility='collapsed')

with col2:
    course_name = st.text_input('Course name')

with col3:
    search_courses_button = st.button('Search courses')
    search_info = st.button('Search info')

language = st.sidebar.radio('Language', ['English', 'Русский'])
cost = st.sidebar.slider("Cost ($)", min_value=0, max_value=500, value=(0, 500))
difficulty = st.sidebar.selectbox("Difficulty", ["Any", "Begginer", "Intermediate", "Advanced"])
rating = st.sidebar.slider("Minimal rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
duration = st.sidebar.slider("Duration (hours)", min_value=1, max_value=100, value=(1, 100))

criteria = {
    'topic': course_name,
    'platform': platform,
    'cost': cost,
    'difficulty': difficulty,
    'rating': rating,
    'duration': duration 
}

if search_courses_button and course_name and platform:
    response = None
    nest_asyncio.apply()
    if platform == 'Stepik':
        response = search_courses_rag(st.session_state.rag_stepik_agent, criteria)
    elif platform == 'Coursera':
        response = search_courses_rag(st.session_state.rag_coursera_agent, criteria)
    else:
        response = search_courses(st.session_state.search_agent, criteria)

    if language == 'Русский':
        response = translate_text(response, language)

    st.write(response)

elif search_info and course_name and platform:
    st.write('nah')

