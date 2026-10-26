
import json
from pathlib import Path

REGISTRY_PATH = Path("data/company_registry.json")


def load_registry():
    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REGISTRY_PATH, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


def register_company(company_id, company_name):
    registry = load_registry()

    if company_id in registry:
        print(f"Company already registered: {company_id}")
        return

    registry[company_id] = {
        "company_name": company_name,
        "financial_documents": [],
        "credit_documents": []
    }

    save_registry(registry)

    print(f"Registered company: {company_name}")


def list_companies():
    registry = load_registry()

    for company_id, details in registry.items():
        print(f"{company_id}: {details['company_name']}")


if __name__ == "__main__":
    register_company(
        "MICROSOFT",
        "Microsoft Corporation"
    )

    register_company(
        "ABC_CORP",
        "ABC Corporation"
    )
    
    
    register_company(
    "AMAZON",
    "AMAZON Corporation"
)

    print("\nRegistered companies:")
    list_companies()
