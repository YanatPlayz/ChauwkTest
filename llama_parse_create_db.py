from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader
import argparse
import os
import shutil
#from llama_index.legacy.readers.json import JSONReader
from dotenv import load_dotenv
from llama_parse import LlamaParse
DB_PATH = "storage"

def main():
    load_dotenv()
    LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

    parser = LlamaParse(
        api_key= LLAMA_CLOUD_API_KEY,
        result_type="markdown"  # "markdown" and "text" are available
    )
    
    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader("./data", file_extractor=file_extractor)
    base_docs = reader.load_data()
    #reader = JSONReader()
    #base_docs = reader.load_data(input_file="./TrainingCenterList.json")
    raw_index = VectorStoreIndex.from_documents(base_docs)
    raw_index.storage_context.persist()

if __name__ == "__main__":
    main()