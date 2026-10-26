
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from company_registry import load_registry

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPANIES_DIR = DATA_DIR / "companies"

DB_PATH = str(DATA_DIR / "chroma_universal")

COLLECTIONS = {
    "financial": "financial_statements",
    "credit": "credit_reports"
}


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def index_documents():
    registry = load_registry()
    embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    for document_type, collection_name in COLLECTIONS.items():
        store = Chroma(
            collection_name=collection_name,
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

        registry_key = f"{document_type}_documents"

        for company_id, company in registry.items():
            for record in company.get(registry_key, []):
                pdf_path = (BASE_DIR / record["path"]).resolve()

                # Only accept registered PDFs inside the expected folder.
                expected_dir = (
                    COMPANIES_DIR / company_id / document_type
                ).resolve()

                if (
                    not pdf_path.is_file()
                    or pdf_path.suffix.lower() != ".pdf"
                    or pdf_path.parent != expected_dir
                ):
                    print(f"Skipping invalid path: {pdf_path}")
                    continue

                file_hash = hashlib.sha256(
                    pdf_path.read_bytes()
                ).hexdigest()

                print(f"\nIndexing {company_id}: {pdf_path.name}")

                documents = PyPDFLoader(str(pdf_path)).load()

                for doc in documents:
                    doc.metadata.update({
                        "company_id": company_id,
                        "document_type": document_type,
                        "file_hash": file_hash,
                        "filename": pdf_path.name
                    })

                chunks = splitter.split_documents(documents)

                # Stable IDs prevent duplicate chunks on reruns.
                prefix = hashlib.sha256(
                    f"{company_id}:{document_type}:{pdf_path.name}".encode()
                ).hexdigest()[:16]

                ids = [
                    f"{prefix}:{file_hash[:16]}:{i}"
                    for i in range(len(chunks))
                ]

                # Remove previously indexed versions of this file.
                existing = store.get(
                    where={
                        "$and": [
                            {"company_id": company_id},
                            {"filename": pdf_path.name}
                        ]
                    }
                )

                if existing["ids"]:
                    store.delete(ids=existing["ids"])

                if chunks:
                    store.add_documents(chunks, ids=ids)

                print(f"Indexed {len(chunks)} chunks")

    print("\nUniversal document indexing completed!")


if __name__ == "__main__":
    index_documents()
