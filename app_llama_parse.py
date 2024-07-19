import streamlit as st
from llama_index.legacy import VectorStoreIndex, ServiceContext, Document
from llama_index.legacy.llms.openai import OpenAI
import openai
from llama_index.legacy import SimpleDirectoryReader
#from llama_index.legacy.readers.json import JSONReader
from dotenv import load_dotenv
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question"}
    ]

@st.cache_resource(show_spinner=False)
def loaddata():
    with st.spinner(text="Loading and indexing the Streamlit docs - hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt="You are an expert on job centres and skill training centres in india and your job is to answer questions related to that. Keep your answers based on facts do not hallucinate features. Answer in bullet points. Answer only from given documents, don't make up answers by yourself."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
index = loaddata()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
