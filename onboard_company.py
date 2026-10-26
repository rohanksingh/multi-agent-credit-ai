
import argparse
import re
import shutil
from pathlib import Path

from company_registry import (
    load_registry,
    register_company
)
from document_registry import register_documents
from document_indexer import index_documents

BASE_DIR = Path(__file__).resolve().parent
COMPANIES_DIR = BASE_DIR / "data" / "companies"


def onboard_company(company_id, company_name, doc_type, pdf_path):
    company_id = company_id.upper().strip()

    if not re.fullmatch(r"[A-Z0-9_]+", company_id):
        raise ValueError(
            "Company ID must contain only A-Z, 0-9 and underscores."
        )

    if doc_type not in ("financial", "credit"):
        raise ValueError("Document type must be financial or credit.")

    source = Path(pdf_path).expanduser().resolve()

    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise FileNotFoundError("A valid PDF file is required.")

    registry = load_registry()

    if company_id in registry:
        if registry[company_id]["company_name"] != company_name:
            raise ValueError(
                "Company ID already belongs to a different name."
            )
    else:
        register_company(company_id, company_name)

    destination = COMPANIES_DIR / company_id / doc_type
    destination.mkdir(parents=True, exist_ok=True)

    target = destination / source.name

    if source != target.resolve():
        if target.exists():
            raise FileExistsError(
                f"Document already exists: {target}"
            )
        shutil.copy2(source, target)

    print(f"\nDocument stored: {target}")

    print("\nUpdating document registry...")
    register_documents()

    print("\nIndexing registered documents...")
    index_documents()

    print(f"\nOnboarding completed for {company_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--company-id", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument(
        "--type",
        choices=["financial", "credit"],
        required=True
    )
    parser.add_argument("--pdf", required=True)

    args = parser.parse_args()

    onboard_company(
        company_id=args.company_id,
        company_name=args.company_name,
        doc_type=args.type,
        pdf_path=args.pdf
    )
