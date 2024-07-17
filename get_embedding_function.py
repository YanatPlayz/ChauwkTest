from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def get_embedding_function():
    embeddings = FastEmbedEmbeddings()
    return embeddings