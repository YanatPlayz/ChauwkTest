from llama_index.legacy import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
reader = SimpleDirectoryReader("./data/tables")
base_docs = reader.load_data()
raw_index = VectorStoreIndex.from_documents(base_docs)
raw_index.storage_context.persist()