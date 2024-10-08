import streamlit as st
from chatbot import RAGChatbot
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Initialize chatbot and load session state for messages
@st.cache_resource
def initialize_chatbot():
    chatbot = RAGChatbot(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        index_name='test',
    )
    return chatbot

chatbot = initialize_chatbot()

st.title("RAG Chatbot")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am Wagner, a highly intelligent and friendly AI assistant. I am developed to provide answers related to Daniel and Daniel's work"}
    ]

# Display chat history with icons for user and bot
def display_chat_messages():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="👤"):  # User icon
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="🤖"):  # Bot icon
                st.markdown(message["content"])

# Call the function to display past messages
display_chat_messages()

# Input prompt from the user, placed below the past messages
prompt = st.chat_input("Ask me anything!")

# If there's a prompt, send it to the chatbot and get the response
if prompt:
    # Add user input to the message history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user message immediately in the chat
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get chatbot response
    response, sources = chatbot.query_chatbot(prompt, k=15, rerank=True, past_messages=st.session_state.messages)

    # Add chatbot response to the message history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display the bot's response immediately after user input
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

# Optionally display relevant documents with metadata
if prompt and sources:
    st.subheader("Relevant Documents")
    if isinstance(sources, list):
        for i, doc in enumerate(sources):
            st.write(f"**Document {i+1} Metadata:**")
            st.json(doc.metadata)
    else:
        st.write(f"**Document 1 Metadata:**")
        st.json({"source": sources})
