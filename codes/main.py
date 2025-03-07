import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ⚠️ Replace this with your actual OpenAI API key
OPENAI_API_KEY = "openai_api_key"
# Function to load movie data from CSV
def load_movie_data(file_path):
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()
    return documents

# Function to create vectorstore for embeddings and retrieval
def build_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore

# Function to set up retrieval-augmented generation (RAG) chain
def setup_retrieval_chain(vectorstore):
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    return retrieval_chain

# Streamlit app setup
st.title("🍿 Movie Recommendation Chatbot")

# Initialize or retrieve chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the retrieval chain only once
@st.cache_resource
def initialize_chain():
    movie_docs = load_movie_data(r"H:/Project and papers work/LLM/New folder (2)/Movie_Recommendation/dataset/train.csv")
    vectorstore = build_vectorstore(movie_docs)
    retrieval_chain = setup_retrieval_chain(vectorstore)
    return retrieval_chain

chain = initialize_chain()

# User input handling
user_question = st.text_input("Ask about movies or recommendations:")
if st.button("Send") and user_question:
    response = chain({"question": user_question, "chat_history": st.session_state.chat_history})
    bot_reply = response["answer"]

    # Update chat history
    st.session_state.chat_history.append((user_question, response["answer"]))

# Display the conversation
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**User:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
