import streamlit as st
from chatbot import RAGChatbot
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def initialize_chatbot():
    # Initialize the chatbot with necessary API keys and settings
    chatbot = RAGChatbot(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        index_name='test',
    )
    return chatbot

chatbot = initialize_chatbot()

# Streamlit app layout
st.title("RAG Chatbot")
st.write("Ask the chatbot anything and get real-time responses.")

# Input prompt from the user
prompt = st.text_input("Enter your prompt:", "")

if prompt:
    # Query the chatbot and get the response
    response = chatbot.query_chatbot(prompt, k=15, rerank=True)

    # Display LLM response
    st.subheader("LLM Response")
    if 'response' in response:
        st.write(response['response'])  # Display the entire response in a readable format

    # Display reranked relevant documents with metadata
    st.subheader("Relevant Documents")
    if 'context_docs' in response:
        reranked_docs = response['context_docs']
        for i, doc in enumerate(reranked_docs):
            st.write(f"**Document {i+1} Metadata:**")
            st.json(doc.metadata)  # Display metadata in JSON format for better structure