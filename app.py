import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from htmlTemplates import css, bot_template, user_template
from streamlit_mic_recorder import mic_recorder
from bhashini_translator import Bhashini #custom module
import base64
from populate_database import load_chunks

# Stored at local paths.
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Language set by the user.
sourceLanguage = "hi"
targetLanguage = "en"

def get_embedding_function():
    """
    Get the FastEmbed embedding function for document embeddings.

    Returns:
        FastEmbedEmbeddings: An instance of FastEmbedEmbeddings for creating document embeddings.
    """
    embeddings = FastEmbedEmbeddings()
    return embeddings

def get_vectorstore():
    """
    Load the vector store from ChromaDB.

    Returns:
        Tuple[Chroma, list[Document]]:
            - Chroma object: The loaded vector store.
            - list[Document]: The document chunks used to create the vector store.
    """
    embedding_function = get_embedding_function()
    chunks = load_chunks()
    if chunks is None:
        st.error("No saved chunks found. Please run populate_database.py first.")
        return None, None
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    st.write("✅ Loaded the database")
    return db, chunks

def get_improved_retriever(vectorstore, chunks):
    """
    Get an advanced retriever with hybrid search.

    Args:
        vectorstore (Chroma): used for semantic search.
        chunks (list[Document]): used for keyword search.

    Returns:
        Ensemble retriever: Improved retriever combining vector and keyword search.
    """
    # Vector store retriever
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Keyword retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def get_conversation_chain(retriever):
    """
    Get a conversational chain using the provided retriever.

    Args:
        retriever (ContextualCompressionRetriever): The retriever to use in the chain.

    Returns:
        ConversationalRetrievalChain: A chain that combines the language model, retriever, and conversation memory.
    """
    system_prompt = """You are a helpful assistant for the Government of India's National Career Service. 
    Provide accurate, concise information about career centers, job opportunities, and related services. 
    Use simple language and give specific details when available. If unsure, say so without making up information."""
    
    human_prompt = """Context: {context}
    
    Human: {question}
    
    Assistant: Let's approach this step-by-step:
    1) First, I'll review the relevant information from the context searching through all of the documents.
    2) Then, I'll provide a clear and concise answer to your question without making up things.
    3) If any details are missing or unclear, I'll mention that.
    
    Here's my response:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

def handle_userinput(user_question):
    """
    Process user input, generate a response, update the chat history, and display results on the Streamlit application.

    This function retrieves Chatbot responses, translates messages, and generates text-to-speech output for the bot's responses.

    Args:
        user_question (str): The user's input question translated to English.
    """
    bhashini = Bhashini("en", sourceLanguage)
    processed_question = f"Make your response accurate and complete by only using information from the provided context. Use addresses. Make the conversation casual but professional. Use bullet points for lengthy responses.  User: {user_question}"

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
    """
    Main function that runs the Streamlit application.

    This function initializes the Streamlit interface, loads the database, and handles user queries and interactions.
    """
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
                retriever = get_improved_retriever(vectorstore, chunks)
                st.session_state.conversation = get_conversation_chain(retriever)

if __name__ == '__main__':
    main()
