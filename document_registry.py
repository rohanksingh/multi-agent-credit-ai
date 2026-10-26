
from pathlib import Path
from company_registry import load_registry, save_registry

COMPANIES_PATH = Path("data/companies")


def register_documents():
    registry = load_registry()

    for company_id, company in registry.items():
        company_folder = COMPANIES_PATH / company_id

        document_types = {
            "financial": "financial_documents",
            "credit": "credit_documents"
        }

        for folder_name, registry_key in document_types.items():
            folder = company_folder / folder_name

            documents = []

            if folder.exists():
                for pdf in sorted(folder.glob("*.pdf")):
                    documents.append({
                        "filename": pdf.name,
                        "path": str(pdf),
                        "document_type": folder_name,
                        "company_id": company_id
                    })

            company[registry_key] = documents

    save_registry(registry)

    return registry


if __name__ == "__main__":
    registry = register_documents()

    for company_id, company in registry.items():
        print(f"\nCompany: {company_id}")

        print(
            "Financial documents:",
            len(company["financial_documents"])
        )

        print(
            "Credit documents:",
            len(company["credit_documents"])
        )
