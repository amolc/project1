
import os
import sys
import hashlib
from pathlib import Path

from pinecone import Pinecone
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# ==================================================
# CONFIGURATION
# ==================================================

INDEX_NAME = "pdf-knowledge"
NAMESPACE = "pdf-documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paragraph-based chunking settings
CHUNK_SIZE = 1200       # Maximum chunk size in characters
CHUNK_OVERLAP = 150     # Overlap between chunks
BATCH_SIZE = 100        # Number of chunks uploaded per batch


# ==================================================
# STEP 1: LOAD PDF
# ==================================================

def load_pdf(pdf_path):
    """Extract PDF pages into LangChain Documents."""

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    # Remove pages containing no extractable text
    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No text found. If this is a scanned PDF, "
            "run OCR before ingestion."
        )

    print(f"PDF: {pdf_path.name}")
    print(f"Pages containing text: {len(documents)}")

    return documents, pdf_path


# ==================================================
# STEP 2: PARAGRAPH-BASED CHUNKING
# ==================================================

def chunk_documents(documents):
    """Split documents using paragraph boundaries first."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentence boundaries
            " ",     # Word boundaries
            "",      # Character fallback
        ],
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# ==================================================
# STEP 3: ADD METADATA AND STABLE IDS
# ==================================================

def prepare_chunks(chunks, pdf_path):
    """Add document metadata and deterministic chunk IDs."""

    document_id = hashlib.sha256(
        str(pdf_path).encode("utf-8")
    ).hexdigest()[:20]

    chunk_ids = []

    for i, chunk in enumerate(chunks):

        # PyMuPDFLoader page numbers are zero-based
        page_number = chunk.metadata.get("page")

        if page_number is not None:
            chunk.metadata["page"] = int(page_number) + 1

        chunk.metadata["source"] = pdf_path.name
        chunk.metadata["document_id"] = document_id
        chunk.metadata["chunk_index"] = i

        # Stable ID for repeat ingestion of the same chunk
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:16]

        chunk_id = (
            f"{document_id}-{i}-{content_hash}"
        )

        chunk_ids.append(chunk_id)

    return chunks, chunk_ids


# ==================================================
# STEP 4: INITIALIZE EMBEDDINGS
# ==================================================

def create_embeddings():
    """Load a local embedding model."""

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# ==================================================
# STEP 5: CONNECT TO PINECONE
# ==================================================

def connect_pinecone():
    """Connect to an existing Pinecone index."""

    api_key = os.environ.get("PINECONE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Please set the PINECONE_API_KEY environment variable."
        )

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    description = index.describe_index()

    expected_dimension = 384

    if description.dimension != expected_dimension:
        raise ValueError(
            f"Index dimension is {description.dimension}. "
            f"Expected {expected_dimension}."
        )

    return index


# ==================================================
# STEP 6: EMBED AND UPLOAD
# ==================================================

def upload_to_pinecone(chunks, chunk_ids, index, embeddings):
    """Generate embeddings and upload chunks in batches."""

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_documents = chunks[start:end]
        batch_ids = chunk_ids[start:end]

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        print(f"Uploaded {end}/{total} chunks")

    print("\nUpload completed successfully.")
    print(f"Namespace: {NAMESPACE}")
    print(f"Total chunks uploaded: {total}")


# ==================================================
# MAIN PIPELINE
# ==================================================

def main(pdf_path):

    documents, pdf_path = load_pdf(pdf_path)

    chunks = chunk_documents(documents)

    chunks, chunk_ids = prepare_chunks(
        chunks,
        pdf_path,
    )

    embeddings = create_embeddings()

    index = connect_pinecone()

    upload_to_pinecone(
        chunks,
        chunk_ids,
        index,
        embeddings,
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python pdf_to_pinecone.py "
            "your_document.pdf"
        )
        sys.exit(1)

    main(sys.argv[1])
