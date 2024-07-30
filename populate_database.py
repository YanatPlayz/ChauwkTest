import argparse
import os
import glob
import shutil
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import UnstructuredFileLoader
import pickle

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"
PARSING_DATA_PATH = "./parsing_data/"
CHUNKS_PATH = "processed_chunks.pkl"
PARSED_FILES_LIST = "parsed_files.json"

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

def main():
    load_dotenv()
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    os.makedirs('llama_parsed', exist_ok=True)
    os.makedirs(PARSING_DATA_PATH, exist_ok=True)

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Update data store.
    documents = load_documents()
    chunks = split_documents(documents)
    save_chunks(chunks, args.reset)
    add_to_chroma(chunks)

def load_parsed_files():
    if os.path.exists(PARSED_FILES_LIST):
        with open(PARSED_FILES_LIST, 'r') as f:
            return json.load(f)
    return []

def save_parsed_files(parsed_files):
    with open(PARSED_FILES_LIST, 'w') as f:
        json.dump(parsed_files, f)

def load_documents():
    parsed_files = load_parsed_files()
    files_to_parse = []

    for file in os.listdir(DATA_PATH):
        if file not in parsed_files:
            files_to_parse.append(file)
            shutil.copy(os.path.join(DATA_PATH, file), PARSING_DATA_PATH)

    if not files_to_parse:
        print("No new files to parse.")
        return load_parsed_documents()
    if (len(files_to_parse) == 1):
        print(f"Parsing {len(files_to_parse)} new file...")
    else:
        print(f"Parsing {len(files_to_parse)} new files...")
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        parsing_instruction="""
            Using the format Column Name : Field Value, convert all of the fields with headers of one row into one single line of text. 
            Example:
                S.No: 1; State / UT: Assam; Location: Guwahati, Employment Exchange; Address: District Employment Exchange, Guwahati AK Azad Road, Rehabari, Guwahati-8; Type Of Center: Employment Exchange.
                S.No: 10; JSS NAME: JSS Ongole; STATE: Andhra Pradesh; DISTRICT: Prakasham; ADDRESS: H.No.3-119/1, Satyanarayanapuram, 2nd Lane, Ongole, MARKAPUR-523002, PRAKASAM (Andhra Pradesh); EMAIL: jss.ongole@gmail.com; MOBILE: 8333046955; COURSES THEY ARE OFFERING: Handicrafts & Carpets; PIN CODE: 523002; Type of Training Centre: JSS.
            Use semicolons to separate different fields.
            """,
        skip_diagonal_text=True
    )
    print("loaded parser")
    file_extractor = {".pdf": parser}
    llama_documents = SimpleDirectoryReader(input_dir=PARSING_DATA_PATH, file_extractor=file_extractor).load_data()
    print("created llama_parse documents")
    with open('llama_parsed/output.md', 'a') as f: 
        for doc in llama_documents:
            f.write(doc.text + '\n')
    
    parsed_files.extend(files_to_parse)
    save_parsed_files(parsed_files)

    for file in os.listdir(PARSING_DATA_PATH):
        os.remove(os.path.join(PARSING_DATA_PATH, file))

    return load_parsed_documents()

def load_parsed_documents():
    loader = UnstructuredFileLoader('llama_parsed/output.md')
    documents = loader.load()
    print("Loaded parsed documents")
    return documents

def save_chunks(chunks: list[Document], reset: bool):
    with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_PATH}")

def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
        return chunks
    else:
        print("❌ No saved chunks found.")
        return None

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # only add documents that don't exist in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # Create IDs, eg: "data/Model_Career_Centres.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)
    output_path = 'llama_parsed/output.md'
    if os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('')
    if os.path.exists(PARSED_FILES_LIST):
        os.remove(PARSED_FILES_LIST)
    if os.path.exists(PARSING_DATA_PATH):
        shutil.rmtree(PARSING_DATA_PATH)
        os.makedirs(PARSING_DATA_PATH)
    print("✨ Cleared database, chunks, parsed files, and output.md")

if __name__ == "__main__":
    main()