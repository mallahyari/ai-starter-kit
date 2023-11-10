from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, UploadFile
from fastapi.responses import StreamingResponse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document, LLMResult
from langchain.chat_models import ChatOllama
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.config import settings
from app.data_models.schemas import UserQuery
from typing import List, Dict, Any, Optional, AsyncGenerator
from langchain.callbacks.base import BaseCallbackHandler


import tempfile
import structlog
import json
import requests
import asyncio
import httpx
import time



logger = structlog.get_logger()

qa_router = r = APIRouter()



model_name = settings.embedding_settings.em_model_name
embedding_dimensions = settings.embedding_settings.embedding_dimensions
model_kwargs = settings.embedding_settings.em_model_kwargs
encode_kwargs = settings.embedding_settings.em_encode_kwargs
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

stream_callback_handler = AsyncIteratorCallbackHandler()


chat_model = ChatOllama(
    # base_url="http://localhost:11434",
    model="llama2",
    verbose=True,
    # callbacks=[stream_callback_handler], 
    # streaming=True,
    callback_manager=CallbackManager([stream_callback_handler]),
)

template = """[INST] <<SYS>> Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. <</SYS>>
{context}
Question: {question}
Helpful Answer:[/INST]"""


QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def initialize_qdrant(host: str, api_key: str, prefer_grpc: bool):
    
    qdrant_client = QdrantClient(host=host, api_key=api_key, prefer_grpc=prefer_grpc)

    def create_collection(collection_name: str):
        try:
            qdrant_client.get_collection(collection_name=collection_name)
            logger.info(f"Collection {collection_name} already exists.")
        except Exception:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE),
                
            ) 
            logger.info(f"Collection {collection_name} is successfully created.")

    create_collection(settings.qdrant_collection_name)
    return qdrant_client

qdrant_client = initialize_qdrant(host=settings.qdrant_host, api_key=settings.qdrant_api_key, prefer_grpc=False)
qdrant_vectordb = Qdrant(qdrant_client, settings.qdrant_collection_name, embedding_model)


qa_chain = load_qa_chain(llm=chat_model, chain_type="stuff", prompt= QA_CHAIN_PROMPT)
qa_chain.callbacks = [stream_callback_handler]

@r.post('/upload')
async def upload_file(request: Request, file: UploadFile):
    filename = file.filename
    status = "success"
    content = await file.read()
    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=settings.text_splitter['chunk_size'], chunk_overlap=settings.text_splitter['chunk_overlap'])
        documents = text_splitter.split_documents(data)
        logger.info(documents[1])
        logger.info(f"Number of Chunks: {len(documents)}")
        
        # Insert the text chunks into the vector database
        insert_into_vectordb(documents, filename)
        
        
    # logger.info(temp_file_path)
    return {"filename": filename, "status": status}


@r.post("/ask")
async def query_index(request: Request, input_query: UserQuery):
    print(input_query)
    question = input_query.question
    relevant_docs = qdrant_vectordb.similarity_search(question, k=1)
    logger.info(f"======{relevant_docs}")
    context = relevant_docs[0].page_content
    filled_prompt = QA_CHAIN_PROMPT.format(question=question, context=context)
    
    request_payload = {
        "model": "llama2",
        "prompt": filled_prompt,
        "format": "json"
    }
    
    try:
        # Send a POST request to the LLM URL
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:11434/api/generate", json=request_payload, timeout=60.0)

            def stream_response_generator():
                for chunk in response.iter_lines():
                    
                    # decoded_chunk = chunk.decode("utf-8")
                    logger.info(f"Received chunk: {chunk}")
                    json_data = json.loads(chunk)
                    # yield json.dumps({"token": chunk['response']}) + "\n"
                        
                    yield json_data['response']
                    time.sleep(1) 
            return StreamingResponse(
                content=stream_response_generator(),
                media_type="text/event-stream"
            )
    except Exception:
        return {"answer": "could not find the answer!"}
        
    

@r.post("/ask1")
async def query_index_another_approach(request: Request, input_query: UserQuery):
    """ This is another function for streaming the response back to client. However, due
        to some issues with langchain library streaming is not done properly. You can
        see more information here: https://github.com/langchain-ai/langchain/issues/13072
    """
    print(input_query)
    question = input_query.question
    relevant_docs = qdrant_vectordb.similarity_search(question, k=1)
    logger.info(f"======{relevant_docs}")
    
    # Set up the callback handler
    stream_callback_handler = AsyncIteratorCallbackHandler()
    gen = create_generator(relevant_docs, question, stream_callback_handler)
    return StreamingResponse(gen, media_type="text/event-stream")
    

async def run_call(relevant_docs, question: str, stream_callback_handler: AsyncIteratorCallbackHandler):
    qa_chain.callbacks = [stream_callback_handler]
    response = await qa_chain.acall({"input_documents": relevant_docs, "question": question})
    # response = await qa_chain.arun(input_documents=relevant_docs, question=question, return_only_outputs=False)
    return response


async def create_generator(relevant_docs, question: str, stream_callback_handler: AsyncIteratorCallbackHandler):
    # run = asyncio.create_task(qa_chain.arun(input_documents=relevant_docs, question=question, return_only_outputs=False))
    run = asyncio.create_task(run_call(relevant_docs, question, stream_callback_handler))  # Await the async task

    async for token in stream_callback_handler.aiter():
        logger.info(token)
        yield token

    await run


def insert_into_vectordb(documents: List[Document], filename: str):
    for d in documents:
        d.metadata["source"] = filename
        # d.metadata["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    qdrant_vectordb.add_documents(documents)
    
async def send_post_request():
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama2",
        "prompt": "What is the capital of united states?"
    }

    headers = {'Content-Type': 'application/json'}

    try:
        logger.info('h---------')
        response = requests.post(url, data=json.dumps(data), headers=headers)
        logger.info(response.json())
        if response.status_code == 200:
            # Request was successful
            response_data = response.json()
            logger.info(response_data)
            return response_data
        else:
            # Request failed
            return {
                "error": f"Request failed with status code {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request exception: {str(e)}"
        }
