import json
import os
import sqlite3
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
DATABASE_PATH = BASE_DIR / "data" / "app.db"
LEGACY_METADATA_PATH = BASE_DIR / "data" / "documents.json"
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70
RETRIEVAL_K = 4

app = Flask(__name__)
index_lock = threading.Lock()
vector_store = None
indexed_signature = None
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)
prompt = ChatPromptTemplate.from_template(
    """Use the context below to answer the question. If the answer is not in
the context, say so clearly. Mention the source filename when useful. No hallucinations

Context:
{context}

Question:
{question}
"""
)


def ensure_storage():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                name TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                modified INTEGER NOT NULL
            )
            """
        )
        if LEGACY_METADATA_PATH.exists():
            existing_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            if existing_count == 0:
                try:
                    legacy_documents = json.loads(
                        LEGACY_METADATA_PATH.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid document metadata in {LEGACY_METADATA_PATH}"
                    ) from exc
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO documents (name, size, modified)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            document["name"],
                            document["size"],
                            document["modified"],
                        )
                        for document in legacy_documents
                    ],
                )


def read_metadata():
    ensure_storage()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT name, size, modified FROM documents ORDER BY name"
        ).fetchall()
    return [dict(row) for row in rows]


def write_metadata(documents):
    ensure_storage()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM documents")
        connection.executemany(
            """
            INSERT INTO documents (name, size, modified)
            VALUES (?, ?, ?)
            """,
            [
                (document["name"], document["size"], document["modified"])
                for document in documents
            ],
        )


def load_file(path):
    if path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(path)).load()
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path)})]


def get_signature(document_records):
    return {
        "documents": [
            (record["name"], record["size"], record["modified"])
            for record in document_records
        ],
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def read_index_manifest():
    if not INDEX_MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def invalidate_index():
    global vector_store, indexed_signature
    vector_store = None
    indexed_signature = None
    if INDEX_DIR.exists():
        for path in INDEX_DIR.iterdir():
            if path.is_file():
                path.unlink()


def save_index(store, signature):
    store.save_local(str(INDEX_DIR))
    INDEX_MANIFEST_PATH.write_text(
        json.dumps(signature, indent=2),
        encoding="utf-8",
    )


def load_saved_index(signature):
    if read_index_manifest() != signature:
        return None
    index_file = INDEX_DIR / "index.faiss"
    store_file = INDEX_DIR / "index.pkl"
    if not index_file.exists() or not store_file.exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_index():
    global vector_store, indexed_signature
    records = read_metadata()
    signature = get_signature(records)
    if vector_store is not None and indexed_signature == signature:
        return vector_store

    with index_lock:
        if vector_store is not None and indexed_signature == signature:
            return vector_store
        saved_index = load_saved_index(signature)
        if saved_index is not None:
            vector_store = saved_index
            indexed_signature = signature
            return vector_store

        documents = []
        for record in records:
            path = DOCUMENTS_DIR / record["name"]
            if path.exists():
                documents.extend(load_file(path))

        if not documents:
            vector_store = None
            indexed_signature = signature
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, embeddings)
        save_index(vector_store, signature)
        indexed_signature = signature
        return vector_store


def answer_query(query):
    store = build_index()
    if store is None:
        raise ValueError("Add at least one document before asking a question.")

    retrieved = store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    ).invoke(query)
    context = "\n\n".join(document.page_content for document in retrieved)
    response = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": query}
    )
    sources = sorted(
        {
            Path(document.metadata.get("source", "Unknown")).name
            for document in retrieved
        }
    )
    return {"answer": response, "sources": sources}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/documents")
def documents():
    return jsonify({"documents": read_metadata()})


@app.post("/api/documents")
def upload_documents():
    uploaded_files = request.files.getlist("files")
    if not uploaded_files or all(not file.filename for file in uploaded_files):
        return jsonify({"error": "Choose at least one PDF, TXT, or MD file."}), 400

    records = read_metadata()
    known_names = {record["name"] for record in records}
    added = []
    for uploaded_file in uploaded_files:
        if not uploaded_file.filename:
            continue
        filename = secure_filename(uploaded_file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {suffix or 'unknown'}"}), 400
        if not filename:
            return jsonify({"error": "One uploaded file has an invalid filename."}), 400

        destination = DOCUMENTS_DIR / filename
        uploaded_file.save(destination)
        stat = destination.stat()
        record = {
            "name": filename,
            "size": stat.st_size,
            "modified": stat.st_mtime_ns,
        }
        if filename not in known_names:
            records.append(record)
            known_names.add(filename)
            added.append(record)
        else:
            for index, existing in enumerate(records):
                if existing["name"] == filename:
                    records[index] = record
                    break

    write_metadata(records)
    invalidate_index()
    return jsonify({"documents": records, "added": added})


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    query_text = str(payload.get("query", "")).strip()
    if not query_text:
        return jsonify({"error": "Enter a question."}), 400
    try:
        return jsonify(answer_query(query_text))
    except (RuntimeError, ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    ensure_storage()
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
