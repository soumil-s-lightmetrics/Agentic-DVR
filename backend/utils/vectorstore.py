import os
from langchain_community.vectorstores.pgvector import PGVector
from utils.embeddings import get_embeddings_function
from utils.pg_connections import get_connection_url_psycopg2
from utils.collection_config import collections

vectorstore = None


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = new_vectorstore()
    return vectorstore


# Creating the vector-store of the collection-name is passed, else using the default collection
def new_vectorstore(collection_name=None):
    return PGVector(
        collection_name=f'{collection_name if collection_name else os.environ["PGVECTOR_COLLECTION_NAME"]}',
        connection_string=get_connection_url_psycopg2(),
        embedding_function=get_embeddings_function(),
    )


vectorstores = {}
for key in collections.keys():
    store = new_vectorstore(collections[key])
    vectorstores[key] = {
        "store": store,
        "retriever": store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.75},
        ),
        "weight": 0.5,
    }
