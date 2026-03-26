import os
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

client = Groq(api_key="YOUR_API_KEY") 

loader = PyPDFLoader("Swapnila_Praharaj_ADE_14032026.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

query = input("Ask a question: ")

retriever = vectorstore.as_retriever()
relevant_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # free, fast, very capable
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)

print("\nAnswer: ")
print(response.choices[0].message.content)