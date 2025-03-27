from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Loading the Text
loader = PyPDFLoader("thebook.pdf")
docs = loader.load()

# Creating the Vector DB 
embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en")
faiss_db = FAISS(embeddings,docs)

# Performing the Hybrid Search on the PDF

# Create BM25 Retriever --> for the Sparse Matrix 
bm25_retriever = BM25Retriever.from_documents(docs)

# Combine both using an Ensemble Retriever --> for the Dense Vector Embeddings
retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_db.as_retriever()], weights=[0.5, 0.5])

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import Ollama

memory = ConversationBufferMemory(memory_key="chat_history")
llm = Ollama(model="mistral")  # Free local model

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

## Backend by the help of the FastAPI

from fastapi import FastAPI
from langchain.llms import Ollama

app = FastAPI()
llm = Ollama(model = "mistral")

@app.get("/ask")
async def ask(question:str):
    stream = llm.stream(question)
    async for response in stream:
        return response