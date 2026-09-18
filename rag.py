import glob
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:4b"

pdf_paths = sorted(glob.glob(r"C:\SRM-Qual\*.pdf"))
if not pdf_paths:
    raise FileNotFoundError("No PDF files were found in C:\\SRM-Qual. Update the path or add PDFs there.")

documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

if not documents:
    raise ValueError("No readable documents were extracted from the PDF files.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(document_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model=LLM_MODEL, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question. If the answer is not in the context, say so clearly.

Context:
{context}

Question:
{question}
"""
)

query = "Build 1 usecase"
context_docs = retriever.invoke(query)
context = "\n\n".join(doc.page_content for doc in context_docs)

response = (prompt | llm | StrOutputParser()).invoke({
    "context": context,
    "question": query,
})

print(response.encode("ascii", "ignore").decode("ascii"))
