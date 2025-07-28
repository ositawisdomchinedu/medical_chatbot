import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# 🌌 Inject custom dark theme & bubble styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    .user-bubble {
        background-color: #22c55e;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: right;
    }
    .assistant-bubble {
        background-color: #38bdf8;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 70%;
        margin-right: auto;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Setup embeddings, retriever, and memory
@st.cache_resource
def load_rag_chain():
    embeddings = download_hugging_face_embeddings()
    index_name = "medi-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chatModel = ChatGroq(model="llama-3.3-70b-versatile")

    # History-aware retriever
    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question."),
        ("human", "{chat_history}"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=chatModel,
        retriever=retriever,
        prompt=history_aware_prompt
    )

    #Prompt for answering questions
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

    #Final retrieval chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

#  Load the chain
rag_chain = load_rag_chain()

#  Streamlit UI
st.set_page_config(page_title="Medical ChatBot", layout="centered")
st.markdown("<h1 style='color:#38bdf8;'>👨🏽‍⚕️ Medical ChatBot</h1>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

#  Chat input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    #  RAG process
    response = rag_chain.invoke({"input": user_input})
    answer = response["answer"]

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
