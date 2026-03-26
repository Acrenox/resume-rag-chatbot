# GenAI Resume Q&A System (RAG)

This project is a Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over PDF documents such as resumes.

## Features
- Query PDF documents using natural language
- Semantic search using vector embeddings
- Context-aware response generation
- Powered by Llama 3 via Groq API

## Tech Stack
- Python
- FAISS (Vector Database)
- HuggingFace Embeddings
- Groq (Llama 3)
- LangChain

## Architecture
1. Load PDF document
2. Split into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS
5. Retrieve relevant chunks
6. Pass context to LLM for answer generation

## How to Run

```bash
pip install -r requirements.txt
python app.py
