from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Function to load movie data from CSV
def load_movie_data(file_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# Function to create vectorstore for retrieval
def build_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key="openai_api_key")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Function to set up retrieval-augmented generation (RAG) chain
def setup_retrieval_chain(vectorstore):
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="openai_api_key")
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return retrieval_chain

# Load data and initialize retrieval chain
movie_docs = load_movie_data(r"H:/Project and papers work/LLM/New folder (2)/Movie_Recommendation/dataset/train.csv")
vectorstore = build_vectorstore(movie_docs)
retrieval_chain = setup_retrieval_chain(vectorstore)

# Request model for chat input
class ChatRequest(BaseModel):
    question: str

# API Endpoint for chat
@app.post("/chat/")
def chat(request: ChatRequest):
    try:
        response = retrieval_chain({"question": request.question, "chat_history": []})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
