from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma 
from pathlib import Path 
from dotenv import load_dotenv 


# 1. Load the api keys 

load_dotenv() 


# 2. Create the LLM and Embedding model 

llm = ChatOpenAI(model = 'gpt-5-mini')
embedder = OpenAIEmbeddings(model = 'text-embedding-3-small', 
                            dimensions= 1024) # defaul 1536 


# 3. Load the documents from source 

loader = DirectoryLoader(path = Path("./cleaned-transcripts/").as_posix(), 
                         loader_cls= TextLoader, 
                         show_progress= True )


docs = loader.load()

# 4. Chunk the text 

chunker = RecursiveCharacterTextSplitter(chunk_size = 500, 
                                         chunk_overlap = 50)


chunks = chunker.split_documents(docs)


# 5. Create a vector store 

vs = Chroma(
    collection_name = 'rag_demo', 
    embedding_function= embedder, 
    persist_directory= Path("./saved-embeddings").as_posix() 
)

# 6. add documents 

vs.add_documents(chunks)


# 7. Retriever 

retriever = vs.as_retriever(search_kwargs = {"k": 3})

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
        ("system", """ You are helpful assistant. Answer the user query 
                        based on the given context only. If you do not know the answer 
                        say I don't know. Do not add any preamble to the response"""), 
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
