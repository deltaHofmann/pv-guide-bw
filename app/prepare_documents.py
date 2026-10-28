
import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.json"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """Entfernt URLs und vereinheitlicht Leerraum."""
    text = re.sub(r"<https?://[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            paragraph_end = text.rfind("\n", start, end)
            if paragraph_end > start + CHUNK_SIZE // 2:
                end = paragraph_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    for path in sorted(RAW_DIR.glob("*.txt")):
        original_text = path.read_text(encoding="utf-8-sig")
        text = clean_text(original_text)
        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            documents.append({
                "source": path.name,
                "chunk_id": index,
                "text": chunk,
            })

        print(
            f"{path.name}: "
            f"{len(original_text):,} → {len(text):,} Zeichen, "
            f"{len(chunks)} Chunks"
        )

    OUTPUT_FILE.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nGesamt: {len(documents)} Chunks")
    print(f"Gespeichert: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
