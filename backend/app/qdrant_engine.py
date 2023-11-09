from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from config import settings
import structlog

logger = structlog.get_logger()

model_name = settings.embedding_settings.em_model_name
embedding_dimensions = settings.embedding_settings.embedding_dimensions
model_kwargs = settings.embedding_settings.em_model_kwargs
encode_kwargs = settings.embedding_settings.em_encode_kwargs
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


class QdrantIndex():

    def __init__(self, qdrant_host: str, qdrant_api_key: str, prefer_grpc: bool):
        self.qdrant_client = QdrantClient(host=qdrant_host, prefer_grpc=prefer_grpc, api_key=qdrant_api_key)
        self.qdrant_vectordb = Qdrant(self.qdrant_client, settings.qdrant_collection_name, embedding_model)
        
        
    def create_collection(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} already exists.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
                
            ) 
            logger.info(f"Collection {collection_name} is successfully created.")

    