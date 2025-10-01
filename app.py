import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with Your PDF", layout="wide")
st.title("Chat with Your Documents 📄")

# --- Core Logic ---
def get_vector_store(file):
    """Processes the uploaded PDF and creates a vector store."""
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the PDF document
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    # Clean up the temporary file
    os.remove(tmp_file_path)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Process the document and create the RAG chain
    with st.spinner("Processing your document..."):
        vector_store = get_vector_store(uploaded_file)
        
        # Set up the Groq LLM
        llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

        # Define a prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context.
            Think step by step before providing a detailed answer. If you don't know the answer, just say that you don't know.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retrieval chain
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        st.success("Document processed! You can now ask questions.")
    
    # User input
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_question})
            st.write(response["answer"])
else:
    st.info("Please upload a PDF file to get started.")