#from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    loader = CSVLoader(file_path='./data/updated_Model_Career_Centres.csv')
    data = loader.load_and_split()
    #print(data[164].metadata['row'])
    add_to_chroma(data[0:165])
    add_to_chroma(data[165:])
    #add_to_chroma(data[165:330])
    #add_to_chroma(data[330:495])
    #add_to_chroma(data[495:660])
    #add_to_chroma(data[660:])

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    #new_chunk_ids = [chunk.metadata["row"] for chunk in chunks]
    db.add_documents(chunks)
    db.persist()

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()