from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma 
from pathlib import Path 
from dotenv import load_dotenv 
from langfuse import get_client 
from dotenv import load_dotenv 

# LOAD THE API KEYS 
load_dotenv() 

langfuse = get_client() 

system_prompt = langfuse.get_prompt(
    name = "rag_app_system_prompt", 
    type= 'text', 
    label= 'staging'
    )



# 1. Load the api keys 

load_dotenv() 
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TRANSCRIPT_DIR = REPO_ROOT / "data" / "processed" 
VECTOR_STORE_DIR = REPO_ROOT / "saved_embedding"

# 2. Create the LLM and Embedding model 

llm = ChatOpenAI(model = 'gpt-5-mini')
embedder = OpenAIEmbeddings(model = 'text-embedding-3-small', 
                            dimensions= 1024) # defaul 1536 


# 3. Load the documents from source 
# chunk_size = 300
# chunk_overlap = 30 
# LOAD AND UPDATE THE KNOWLEDGE BASE
def upsert_document(chunk_size: int, chunk_overlap: int): 

    # LOAD THE DOCS FROM THE SOURCE 

    loader = DirectoryLoader(path = PROCESSED_TRANSCRIPT_DIR.as_posix(), 
                            loader_cls= TextLoader, 
                            show_progress= True )


    docs = loader.load()

    # 4. Chunk the text 

    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                                encoding_name= "o200k_base",
                                                chunk_size = chunk_size, 
                                            chunk_overlap = chunk_overlap)


    chunks = chunker.split_documents(docs)


    # 5. Create a vector store 

    vs = Chroma(
        collection_name = 'rag_demo', 
        embedding_function= embedder, 
        persist_directory= VECTOR_STORE_DIR.as_posix() 
    )

    # 6. add documents 

    vs.add_documents(chunks)

    return vs 


def load_knowledge_base(): 
    # vector_store 
    vs = Chroma(
        collection_name= "rag_demo", 
        embedding_function= embedder, 
        persist_directory= VECTOR_STORE_DIR.as_posix()
    )

    return vs 

if VECTOR_STORE_DIR.exists(): 
    vs = load_knowledge_base() 
else: 
    vs = upsert_document(chunk_size, chunk_overlap)


# 7. Retriever 

retriever = vs.as_retriever(search_type = "similarity", 
                            search_kwargs = {"k": 3})

# creating langgraph 
from langgraph.graph import StateGraph , START, END 
from typing import TypedDict, Literal

# 8. Creating langgraph graph 

class RagState(TypedDict): 
    query: str 
    retrieved_docs : list[Document]
    context: str 
    prompt: ChatPromptTemplate
    response: str 

def retrieve(state: RagState) -> dict: 
    query = state['query']
    retrieved_docs = retriever.invoke(query)


    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    return {"retrieved_docs": retrieved_docs, 
            "context": context}

def augmentation(state: RagState) -> dict: 
    # query = state['query']
    # context = state['context']

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.prompt), 
        ("human", "context: {context}\n\nquery: {query}")
    ])

    return {"prompt": prompt}

from langchain_core.output_parsers import StrOutputParser

def generation(state: RagState)-> dict: 

    query = state['query']
    context = state['context']
    prompt = state['prompt']

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": context, 
                                 "query": query})

    return {"response": response}

graph_builder = StateGraph(RagState) 

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)


graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "augmentation")
graph_builder.add_edge("augmentation", "generation")
graph_builder.add_edge("generation", END)


graph = graph_builder.compile()
