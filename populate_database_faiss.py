#Using FAISS to populate
#Hard coded for now
#Use the updated_filename.csv for loading the csv as it is formatted correctly
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import FAISS
import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

CHROMA_PATH = "faiss_index_2"
DATA_PATH = "data/D"
EMBED = embed = get_embedding_function()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    loader = CSVLoader(file_path='./data/updated_Model_Career_Centres.csv')
    data = loader.load_and_split()
    db = FAISS.from_documents(data, EMBED)
    db.save_local(CHROMA_PATH)

    loader = CSVLoader(file_path='./data/updated_PMKKs.csv')
    data = loader.load_and_split()
    db = FAISS.from_documents(data, EMBED)
    saved_db = FAISS.load_local(CHROMA_PATH, EMBED, allow_dangerous_deserialization=True)
    saved_db.merge_from(db)

    documents = load_documents()
    chunks = split_documents(documents)
    db_pdf = FAISS.from_documents(chunks, EMBED)
    saved_db.merge_from(db_pdf)

    show_vstore(saved_db)

def show_vstore(store) :
    vector_df = store_to_df(store)
    print(vector_df["document"])

def store_to_df(store):
    v_dict = store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source'].split('/')[-1]
        content = v_dict[k].page_content
        data_rows.append({"chunk_id":k, "document" :doc_name, "content": content})
    vector_df = pd.DataFrame(data_rows)
    return vector_df

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    main()