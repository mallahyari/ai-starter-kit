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

template = """[INST] <<SYS>> When users upload a document, please disregard the example passage in this prompt. Extract remarkable or interesting points directly from the user's document. Act as a content marketing assistant and help the user find "purple cow" worthy points to tweet about based on the content they provide.

Ensure that the examples you generate are specific to the user-uploaded document and not repetitive. Aim for creativity, uniqueness, and a focus on insights that would resonate with innovators and forward-thinking individuals. 

Do not provide any points from the example passage below between ``` and ```

>>>> 9 examples of Purple cow points from the example passage below:

1. three qualities to decide what to work on: it has to be something you have a natural aptitude for, that you have a deep interest in, and that offers scope to do great work
2. That's easier said than done, you have to "figure it out". And the way is to Develop a habit of working on your own projects. Don't let "work" mean something other people tell you to do. 
3. the ultimate driver is gonna be "excited curiosity". There's a kind of excited curiosity that's both the engine and the rudder of great work. It will not only drive you, but if you let it have its way, will also show you what to work on.
4. Knowledge expands fractally, and from a distance its edges look smooth, but once you learn enough to get close to one, they turn out to be full of gaps.
5. The next step is to notice them. Many discoveries have come from asking questions about things that everyone else took for granted.
6. Great work often has a tincture of strangeness. 
7. Boldly chase outlier ideas, even if other people aren't interested in them — in fact, especially if they aren't.
8. Four steps: choose a field, learn enough to get to the frontier, notice gaps, explore promising ones. This is how practically everyone who's done great work has done it, from painters to physicists.
9. The three most powerful motives are curiosity, delight, and the desire to do something impressive.
Do you now understand what I think is interesting and remarkable in the context?

```
>>>>> example passage:
“If you collected lists of techniques for doing great work in a lot of different fields, what would the intersection look like? I decided to find out by making it.

Partly my goal was to create a guide that could be used by someone working in any field. But I was also curious about the shape of the intersection. And one thing this exercise shows is that it does have a definite shape; it's not just a point labelled "work hard."

The following recipe assumes you're very ambitious.

The first step is to decide what to work on. The work you choose needs to have three qualities: it has to be something you have a natural aptitude for, that you have a deep interest in, and that offers scope to do great work.

In practice you don't have to worry much about the third criterion. Ambitious people are if anything already too conservative about it. So all you need to do is find something you have an aptitude for and great interest in. [1]

That sounds straightforward, but it's often quite difficult. When you're young you don't know what you're good at or what different kinds of work are like. Some kinds of work you end up doing may not even exist yet. So while some people know what they want to do at 14, most have to figure it out.

The way to figure out what to work on is by working. If you're not sure what to work on, guess. But pick something and get going. You'll probably guess wrong some of the time, but that's fine. It's good to know about multiple things; some of the biggest discoveries come from noticing connections between different fields.

Develop a habit of working on your own projects. Don't let "work" mean something other people tell you to do. If you do manage to do great work one day, it will probably be on a project of your own. It may be within some bigger project, but you'll be driving your part of it.

What should your projects be? Whatever seems to you excitingly ambitious. As you grow older and your taste in projects evolves, exciting and important will converge. At 7 it may seem excitingly ambitious to build huge things out of Lego, then at 14 to teach yourself calculus, till at 21 you're starting to explore unanswered questions in physics. But always preserve excitingness.

There's a kind of excited curiosity that's both the engine and the rudder of great work. It will not only drive you, but if you let it have its way, will also show you what to work on.

What are you excessively curious about — curious to a degree that would bore most other people? That's what you're looking for.

Once you've found something you're excessively interested in, the next step is to learn enough about it to get you to one of the frontiers of knowledge. Knowledge expands fractally, and from a distance its edges look smooth, but once you learn enough to get close to one, they turn out to be full of gaps.

The next step is to notice them. This takes some skill, because your brain wants to ignore such gaps in order to make a simpler model of the world. Many discoveries have come from asking questions about things that everyone else took for granted. [2]

If the answers seem strange, so much the better. Great work often has a tincture of strangeness. You see this from painting to math. It would be affected to try to manufacture it, but if it appears, embrace it.

Boldly chase outlier ideas, even if other people aren't interested in them — in fact, especially if they aren't. If you're excited about some possibility that everyone else ignores, and you have enough expertise to say precisely what they're all overlooking, that's as good a bet as you'll find. [3]

Four steps: choose a field, learn enough to get to the frontier, notice gaps, explore promising ones. This is how practically everyone who's done great work has done it, from painters to physicists.

Steps two and four will require hard work. It may not be possible to prove that you have to work hard to do great things, but the empirical evidence is on the scale of the evidence for mortality. That's why it's essential to work on something you're deeply interested in. Interest will drive you to work harder than mere diligence ever could.

The three most powerful motives are curiosity, delight, and the desire to do something impressive. Sometimes they converge, and that combination is the most powerful of all.

The big prize is to discover a new fractal bud. You notice a crack in the surface of knowledge, pry it open, and there's a whole world inside.”
```


If you don't know the answer, just say that you don't know, don't try to make up an answer. 
 <</SYS>>
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
