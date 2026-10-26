
from langchain_chroma import Chroma

from document_indexer import (
    DB_PATH,
    COLLECTIONS,
    get_embeddings
)

embeddings = get_embeddings()

store = Chroma(
    collection_name=COLLECTIONS["financial"],
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

questions = [
    "Amazon consolidated balance sheet total assets liabilities debt",
    "Amazon consolidated statement of cash flows operating cash flow",
    "Amazon consolidated statement of operations revenue operating income"
]

for question in questions:
    print("\nQUESTION:", question)

    docs = store.similarity_search(
        question,
        k=5,
        filter={
            "$and": [
                {"company_id": {"$eq": "AMAZON"}},
                {"document_type": {"$eq": "financial"}}
            ]
        }
    )

    for doc in docs:
        print(
            "Document:", doc.metadata.get("filename"),
            "| Page:", doc.metadata.get("page", 0) + 1
        )
