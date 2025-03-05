#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:02:35 2025

@author: megha
"""

import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables
load_dotenv()

# Get OpenAI API Key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ChromaDB persistence directory
CHROMA_DB_DIR = "chromadb"

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load the persisted ChromaDB
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)

# Define LLM Model
llm_name = "gpt-4"

# Define Custom QA Prompt
QA_TEMPLATE = """You are a helpful support bot for Hopstack providing information about Fulfillment By Amazon.
Use only the provided documentation to answer the question.
Do NOT make up answers—if you are unsure, simply say you don't know.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["question", "context"])

# Define Retriever & Memory
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

# Initialize Conversation Chain (Only Once)
if "chain" not in st.session_state:
    st.session_state["chain"] = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0, max_tokens=2048),
        memory=memory,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        rephrase_question=False,
        condense_question_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    )

# Initialize Chat History
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hello! Ask me anything about Fulfillment by Amazon."]

if "past" not in st.session_state:
    st.session_state["past"] = ["Hi!"]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Function to clear text input
def clear_text():
    st.session_state.user_input = st.session_state.input
    st.session_state.input = ""

# Streamlit UI
st.title("FBA Help Chatbot")
response_container = st.container()
container = st.container()

with container:
    st.text_input(" ", placeholder="Ask me anything about Amazon Fulfillment", key="input", on_change=clear_text)

    if st.session_state.user_input:
        output = st.session_state.chain({"question": st.session_state.user_input})
        output = output["answer"]
        chat_history = st.session_state["chat_history"]

        st.session_state["past"].append(st.session_state.user_input)
        st.session_state["generated"].append(output)
        st.session_state.chat_history.append(chat_history)

# Display Chat History
if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            with st.chat_message("user", avatar="https://raw.githubusercontent.com/megha-hopstack/streamlit-semanticsearch/main/person.png"):
                st.markdown(st.session_state["past"][i])
            with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/megha-hopstack/streamlit-semanticsearch/main/hopstacklogo.png"):
                st.markdown(st.session_state["generated"][i])

# Hide Streamlit Default UI
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
