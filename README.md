# 🩺 AI Medical Assistant Chatbot

A RAG-based medical chatbot that allows users to upload medical PDFs and ask domain-specific questions using LLM-powered retrieval and generation.

---

## 🚀 Features

* Upload medical PDFs
* Semantic document chunking
* Vector search using Pinecone
* RAG pipeline with LangChain
* LLaMA3-70B inference via Groq
* FastAPI backend APIs
* Streamlit frontend interface

---

## 🧠 Architecture

```text
User Query
   -> Embedding Generation
   -> Pinecone Vector Search
   -> Relevant Context Retrieval
   -> RAG Chain (LangChain + Groq)
   -> LLM Response
```

---

## 🛠️ Tech Stack

### AI / ML

* LangChain
* Groq (LLaMA3-70B)
* Pinecone
* Google Generative AI Embeddings

### Backend

* FastAPI
* Python

### Frontend

* Streamlit

### Deployment

* Render

---

## 📂 Project Structure

```text
client/      -> Streamlit frontend
server/      -> FastAPI backend
assets/      -> PDFs and architecture docs
```

---

## ⚡ Core Functionalities

* PDF upload and processing
* Text extraction and chunking
* Embedding generation
* Vector similarity search
* Context-aware medical Q&A

---

## 📌 API Endpoints

```text
POST /upload_pdfs/
POST /ask/
```

---

## 💡 Highlights

* Reduced hallucinations using Retrieval-Augmented Generation
* Built scalable vector-search pipeline for medical documents
* Integrated modern LLM stack with production-ready APIs
* Designed modular backend and frontend architecture

---

