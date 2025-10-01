# 📄 Chat with Your PDF – A RAG-based Chatbot  

## 🚀 Overview  
An interactive web application that allows users to upload a PDF and ask questions about its content.  
It uses a **Retrieval-Augmented Generation (RAG) pipeline** to fetch relevant context from the document and generate accurate answers using large language models.  

## ✨ Features  
- 📂 Upload any PDF and instantly query its contents.  
- 💬 Ask questions in plain English and get context-aware answers.  
- ⚡ Real-time processing with embeddings and vector search.  
- 🤖 Powered by LLMs (Groq API + Hugging Face embeddings).  
- 🖥️ Simple and intuitive Streamlit interface.  

## 🛠️ Tech Stack  
- **Language:** Python  
- **Frameworks:** LangChain, Streamlit  
- **LLM Backend:** Groq API (Meta Llama 3.1)  
- **Embeddings:** Hugging Face (`sentence-transformers`)  
- **Vector Store:** FAISS (in-memory)  
- **Environment:** Conda  

## 🔮 Future Upgradation Roadmap  
- Add conversational memory for follow-up queries.  
- Deploy publicly (Streamlit Community Cloud).  
- Extend support to `.txt` and `.docx`.  
- Experiment with hybrid retrieval (BM25 + embeddings).  

---

 *Built as part of my journey in LLM-powered applications and document intelligence.*  
