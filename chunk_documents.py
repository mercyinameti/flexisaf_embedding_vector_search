# chunk_documents.py
# I split each document into paragraph chunks and save them with their metadata.

import csv
import json

REGISTER_FILE = "sources.csv"
DOCUMENTS_FOLDER = "documents"
OUTPUT_FILE = "chunks.json"


def split_paragraphs(text):
    # I split the text wherever there is a blank line, and drop empty pieces.
    blocks = text.replace("\r\n", "\n").split("\n\n")
    return [block.strip() for block in blocks if block.strip()]


def main():
    chunks = []
    with open(REGISTER_FILE, encoding="utf-8", newline="") as register:
        for source in csv.DictReader(register):
            path = f"{DOCUMENTS_FOLDER}/{source['file_name']}"
            with open(path, encoding="utf-8") as file:
                blocks = split_paragraphs(file.read())

            # I skip the first block, because it is the document title.
            body = blocks[1:]
            for number, paragraph in enumerate(body, start=1):
                chunks.append({
                    "chunk_id": f"{source['doc_id']}-{number:03d}",
                    "doc_id": source["doc_id"],
                    "title": source["title"],
                    "category": source["category"],
                    "date_written": source["date_written"],
                    "text": paragraph,
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()