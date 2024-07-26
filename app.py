import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from htmlTemplates import css, bot_template, user_template
from streamlit_mic_recorder import mic_recorder
from bhashini_translator import Bhashini
import base64
from populate_database import load_documents, split_documents

CHROMA_PATH = "chroma"
DATA_PATH = "data"
sourceLanguage = "hi"
targetLanguage = "en"

def get_vectorstore():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    documents = load_documents()
    chunks = split_documents(documents)
    st.write("✅ Loaded the database!")
    return db, chunks

def get_conversation_chain(vectorstore, chunks):
    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 2
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.05, 0.95]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory
    )
    
    return conversation_chain


def handle_userinput(user_question):
    bhashini = Bhashini("en", sourceLanguage)

    processed_question = f"Use addresses and make sure they are correct. Try to use simpler and more common words. Make the conversation casual but professional. Use bullet points for lengthy responses. User: {user_question}"

    response = st.session_state.conversation({'question': processed_question})
    st.session_state.chat_history = response['chat_history']
    chat_history = response['chat_history']
    translated_new_messages = []

    for index, message in enumerate(chat_history):
        message_id = str(index)
        if message_id not in st.session_state.translated_messages_record:
            st.session_state.translated_messages_record.add(message_id)
            if hasattr(message, 'content'):
                if index % 2 == 0:
                    st.write("Recieving message with original prompt, will cut for translation: ", message)
                    user_question_content = getattr(message, 'content').split("User: ")[1]
                    translated_message_content = bhashini.translate(user_question_content)
                    st.write("User message, no need for audio: ", user_question_content)
                    base64_aud = ""
                else:
                    bot_question_content = getattr(message, 'content')
                    translated_message_content = bhashini.translate(bot_question_content)
                    st.write("Bot response, generating audio: ", bot_question_content)
                    bhashini2 = Bhashini(sourceLanguage, targetLanguage)
                    base64_aud = bhashini2.tts(translated_message_content)

                translated_new_messages.append({
                    'text': translated_message_content,
                    'audio': base64_aud
                })

    st.session_state.translated_chat_history.extend(translated_new_messages)
    for i, message_data in enumerate(st.session_state.translated_chat_history):
        translated_message = message_data['text']
        base64_aud = message_data['audio']

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", translated_message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", translated_message), unsafe_allow_html=True)
            st.audio(base64.b64decode(base64_aud), format="audio/wav")
            
def main():
    load_dotenv()
    st.set_page_config(page_title="ChauwkBot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "translated_chat_history" not in st.session_state:
        st.session_state.translated_chat_history = []
    if "translated_messages_record" not in st.session_state:
        st.session_state.translated_messages_record = set()

    st.header("Chauwk Bot")
    user_question = st.text_input("Ask away!")

    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", format="wav")
    with send_button_column:
        send_button = st.button("Send", key="send_button")

    if send_button:
        if user_question:
            bhashini = Bhashini(sourceLanguage, "en")
            user_question = bhashini.translate(user_question)
            handle_userinput(user_question)
            user_question = None
        elif voice_recording:
            audio_base64_string = base64.b64encode(voice_recording['bytes']).decode('utf-8')
            bhashini = Bhashini(sourceLanguage, targetLanguage)
            text = bhashini.asr_nmt(audio_base64_string)
            user_question = text
            handle_userinput(user_question)
            voice_recording = None

    if not user_question:
        user_question = None
        voice_recording = None

    with st.sidebar:
        if st.button("Load Database"):
            with st.spinner("Loading"):
                vectorstore, chunks = get_vectorstore()
                st.session_state.conversation = get_conversation_chain(vectorstore, chunks)

if __name__ == '__main__':
    main()
