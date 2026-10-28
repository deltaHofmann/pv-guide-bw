import json
from pathlib import Path
from ollama import embed


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "chunks.json"
INDEX_FILE = PROJECT_DIR / "data" / "processed" / "search_index.json"

EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 16


def load_chunks():
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

def build_index(chunks):
    index = []

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        response = embed(model=EMBEDDING_MODEL,input=texts,)

        for chunk, embedding in zip(batch, response["embeddings"]):
            index.append({**chunk, "embedding": embedding,})

        print(f"Verarbeitet: {len(index)} / {len(chunks)}")
    return index

def save_index(index):
    with INDEX_FILE.open("w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False)

    print(f"Suchindex gespeichert: {INDEX_FILE}")

def main():
    chunks = load_chunks()
    if not chunks:
        print("Keine Textabschnitte gefunden.")
        return
    
    index = build_index(chunks)
    save_index(index)

if __name__ == "__main__":
    main()
