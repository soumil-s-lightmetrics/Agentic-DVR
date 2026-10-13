from langchain_openai import OpenAIEmbeddings

embeddings = None


def get_embeddings_function():
    global embeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings()
    return embeddings
