
from pathlib import Path

from company_registry import load_registry
from document_indexer import DB_PATH, COLLECTIONS
from langchain_chroma import Chroma


def check_document_availability(company_id):
    registry = load_registry()

    if company_id not in registry:
        return {
            "company_id": company_id,
            "financial": False,
            "credit": False
        }

    availability = {
        "company_id": company_id,
        "financial": False,
        "credit": False
    }

    for doc_type, collection_name in COLLECTIONS.items():

        db_file = Path(DB_PATH) / "chroma.sqlite3"

        if not db_file.exists():
            continue

        store = Chroma(
            collection_name=collection_name,
            persist_directory=DB_PATH
        )

        records = store.get(
            where={"company_id": company_id},
            limit=1
        )

        availability[doc_type] = bool(records["ids"])

    return availability


if __name__ == "__main__":
    for company in ["MICROSOFT", "ABC_CORP", "AMAZON"]:
        result = check_document_availability(company)

        print(f"\nCompany: {company}")
        print("Financial documents available:", result["financial"])
        print("Credit documents available:", result["credit"])
