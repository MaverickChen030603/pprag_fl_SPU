import json
from pathlib import Path

from llama_index.core import Document


DATA_DIR = Path(__file__).resolve().parent


def _load_optional_test_corpus(documents):
    test_corpus_path = DATA_DIR / "test_corpus.json"
    if not test_corpus_path.exists():
        return
    with test_corpus_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    count = 0
    for _, entry in data.items():
        for _, passage in entry.items():
            text = passage["page_content"]
            doc_id = passage["index"]
            document = Document(text=text, metadata={"title": "", "id": doc_id}, doc_id=str(doc_id))
            documents.append(document)
            count += 1
            if count == 6066:
                return


def _load_reference_dataset(documents):
    for candidate in ("data_50.json", "data_100.json"):
        path = DATA_DIR / candidate
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        for entry in data:
            title = entry["other_info"]["doc_name"]
            for reference, ref_id in zip(entry["key_content"]["reference"], entry["key_content"]["reference_idx"]):
                document = Document(text=reference, metadata={"title": title, "id": ref_id}, doc_id=str(ref_id))
                documents.append(document)
        return
    raise FileNotFoundError("No reference dataset found in RAGTest/data. Expected data_50.json or data_100.json.")


def get_documents():
    documents = []
    _load_optional_test_corpus(documents)
    _load_reference_dataset(documents)
    print("len(documents):", len(documents))
    return documents


if __name__ == "__main__":
    documents = get_documents()
    print(documents)
