
import json
import re
from pathlib import Path

INPUT_PATH = Path("data/smu_public_pages.jsonl")
OUTPUT_PATH = Path("data/chunks.jsonl")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_PAGE_LENGTH = 500

def normalise_for_deduplication(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()

def split_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(". ", start, end)

            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
            else:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - CHUNK_OVERLAP)

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def load_pages() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}. "
            "Run python -m src.ingest_web first."
        )

    pages = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}.")
                continue

            text = record.get("text", "").strip()
            metadata = record.get("metadata", {})

            if len(text) < MIN_PAGE_LENGTH:
                title = metadata.get("title", "Unknown page")
                print(f"Skipping short page: {title}")
                continue

            if not metadata.get("source_url"):
                print("Skipping page without a source URL.")
                continue

            pages.append(record)

    return pages

def build_chunks(pages: list[dict]) -> list[dict]:
    all_chunks = []
    seen_chunks = set()

    for page in pages:
        metadata = page["metadata"]
        page_chunks = split_text(page["text"])
        added_from_page = 0

        for index, chunk_text in enumerate(page_chunks):
            normalised = normalise_for_deduplication(chunk_text)

            if normalised in seen_chunks:
                continue

            seen_chunks.add(normalised)

            all_chunks.append(
                {
                    "chunk_id": f"chunk_{len(all_chunks) + 1:06d}",
                    "text": chunk_text,
                    "metadata": {
                        "title": metadata.get("title", "Unknown"),
                        "source_url": metadata["source_url"],
                        "retrieved_at": metadata.get("retrieved_at"),
                        "chunk_index": index,
                    },
                }
            )

            added_from_page += 1

        print(
            f"{metadata.get('title', 'Unknown')}: "
            f"{added_from_page} unique chunks"
        )

    return all_chunks

def main():
    pages = load_pages()
    chunks = build_chunks(pages)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nPages processed: {len(pages)}")
    print(f"Unique chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["metadata"])
        print(chunks[0]["text"][:300])

if __name__ == "__main__":
    main()
