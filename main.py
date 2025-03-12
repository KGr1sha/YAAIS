import streamlit as st
import asyncio
import aiohttp
import google.generativeai as genai
from dotenv import load_dotenv
import os


def init_session():
    st.session_state["initialized"] = True;
    st.session_state.messages = [];
    st.session_state.prompt = "";
    load_dotenv();
    key = os.getenv("GENIMI_API_KEY");
    genai.configure(api_key=key);


def fetch_ai_response(prompt, model="gemini-1.5-flash"):
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


# TODO(grisha): do multiple chat sessions. database angle?
#st.sidebar.title("Chat History")
#for msg in st.session_state.messages:
#    st.sidebar.write(f"**You:** {msg['user']}")
#    st.sidebar.write(f"**AI:** {msg['model']}")

st.title("AI is EVIL")

# Display chat messages with alignment
for msg in st.session_state.messages:
    col1, col2 = st.columns([1, 5], border=False);
    with col1:
        st.write("");

    with col2:
        message = st.chat_message("user");
        message.write(f"{msg['user']}");

    col1, col2 = st.columns([5, 1], border=False);

    with col1:
        message = st.chat_message("ai");
        message.write(f"{msg['model']}");

    with col2:
        st.write("");

st.divider();
prompt = st.chat_input("Ask about anything")
if prompt:
    ai_response = fetch_ai_response(prompt);
    st.session_state.messages.append({"user": prompt, "model": ai_response});
    st.rerun();
