
import json
import math
import re
from pathlib import Path

from ollama import embed


PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "data" / "processed" / "search_index.json"

EMBEDDING_MODEL = "nomic-embed-text"
STOPWORDS = {
    "wie", "was", "wann", "wo", "wer", "welche", "welcher", "welches",
    "ist", "sind", "wird", "werden", "gilt", "gelten", "für", "bei",
    "mit", "von", "auf", "aus", "und", "oder", "eine", "einer",
    "einem", "einen", "der", "die", "das", "den", "dem", "zu", "im",
    "in", "es", "sich", "muss", "müssen", "kann", "können",
}


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def keywords(text):
    words = re.findall(r"\b[a-zäöüß0-9]+\b", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def keyword_score(question, text):
    query_words = keywords(question)
    if not query_words:
        return 0.0

    return len(query_words & keywords(text)) / len(query_words)


def search(question, index, top_k=5):
    response = embed(model=EMBEDDING_MODEL, input=question)
    question_vector = response["embeddings"][0]
    results = []

    for chunk in index:
        semantic = cosine_similarity(question_vector, chunk["embedding"])
        lexical = keyword_score(question, chunk["text"])
        combined = 0.75 * semantic + 0.25 * lexical
        results.append((combined, semantic, lexical, chunk))

    results.sort(key=lambda item: item[0], reverse=True)
    return results[:top_k]


def main():
    with INDEX_FILE.open("r", encoding="utf-8") as file:
        index = json.load(file)

    print(f"Suchindex geladen: {len(index)} Abschnitte")
    print("Stelle eine Frage zur Photovoltaikpflicht in BW.")
    print("Mit 'exit' beendest du das Programm.")

    while True:
        question = input("\nDeine Frage: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        for rank, (combined, semantic, lexical, chunk) in enumerate(search(question, index), start=1):
            print(
                f"\n--- Treffer {rank} | Gesamt: {combined:.3f}"
                f" | Semantik: {semantic:.3f} | Keywords: {lexical:.3f} ---"
            )
            print(f"Quelle: {chunk.get('source', 'unbekannt')}")
            print(chunk.get("text", "")[:1200])


if __name__ == "__main__":
    main()
