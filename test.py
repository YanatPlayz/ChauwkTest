# To check working of Unstructed library
#Result - Too slow and is not giving correct responses

import argparse
import os
import shutil
#from langchain.document_loaders.pdf import PyPDFDirectoryLoader
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    load_dotenv()
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    client = UnstructuredClient(
        api_key_auth="LUUVEhLj3PQ2fQ4x9c4RaryyLmBUEr",
    )
    path_to_pdf="./data/List_of_PMKKs.pdf"
    with open(path_to_pdf, "rb") as f:
        files=shared.Files(
            content=f.read(),
            file_name=path_to_pdf,
            )
        req = shared.PartitionParameters(
            files=files,
            chunking_strategy="by_title",
            max_characters=512,
        )
        try:
            resp = client.general.partition(req)
        except SDKError as e:
            print(e)
    elements = dict_to_elements(resp.elements)
    documents = []
    for element in elements:
        metadata = element.metadata.to_dict()
        metadata['languages'] = 'eng'
        documents.append(Document(page_content=element.text, metadata=metadata))
    print(documents)
    add_to_chroma(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    for doc in chunks:
        db.add_texts(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()