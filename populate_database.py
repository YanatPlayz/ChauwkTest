import argparse
import os
import glob
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

CHROMA_PATH = "chroma"
DATA_PATH = "./data/"

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

def main():
    load_dotenv()
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    os.makedirs('llama_parsed', exist_ok=True)
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Update data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    print("loading documents")
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        parsing_instruction="The provided documents contain many tables with addresses. If the state value is empty, that row refers to the state from the row before, be sure to add that state to that row while parsing. Be precise in extracting information.",
        skip_diagonal_text=True
    )
    print("loaded parser")
    file_extractor = {".pdf": parser}
    llama_documents = SimpleDirectoryReader(input_dir=DATA_PATH, file_extractor=file_extractor).load_data()
    print("created llama_parse documents")
    with open('llama_parsed/output.md', 'a') as f: 
        for doc in llama_documents:
            f.write(doc.text + '\n')
    loader = DirectoryLoader('llama_parsed/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    print("initialized final documents")
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
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
    output_path = 'llama_parsed/output.md'
    if os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('')
        print("✨ Cleared output.md")


if __name__ == "__main__":
    main()