#from dotenv import load_dotenv
from llama_index.legacy import StorageContext, load_index_from_storage
from llama_index.legacy.postprocessor import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

storage_context = StorageContext.from_defaults(persist_dir="./storage")
raw_index = load_index_from_storage(storage_context)
raw_query_engine = raw_index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[reranker]
)
query="address of training centres for electrician in THRISSUR city"
print(raw_query_engine.query(query)) 