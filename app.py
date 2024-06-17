import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
from htmlTemplates import css, bot_template, user_template
from streamlit_mic_recorder import mic_recorder

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def get_vectorstore():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    st.write("✅ Loaded the database!")
    return db

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def translate(audio):
    return

def main():
    load_dotenv()
    st.set_page_config(page_title="ChauwkBot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChauwkBot (multiple PDFs)")
    user_question = st.text_input("Ask a question about your documents:")

    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording=mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording")
    with send_button_column:
        send_button = st.button("Send", key="send_button")
    
    if voice_recording:
        translation = translate(voice_recording["bytes"])
    
    print(voice_recording)
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        if st.button("Load Database"):
            with st.spinner("Loading"):
                vectorstore = get_vectorstore()
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()