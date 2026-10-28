
import json
from pathlib import Path
from ollama import chat
from search import search


PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = PROJECT_DIR / "data" / "processed" / "search_index.json"

LLM_MODEL = "qwen3.5:4b"
TOP_K = 5


def load_index():
    with INDEX_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

def build_context(results):
    context_parts = []
    sources = []

    for number, (_, _, _, chunk) in enumerate(results, start=1):
        source_name = chunk.get("source", "Unbekannte Quelle")
        context_parts.append(f"[Quelle {number}: {source_name}]\n{chunk.get('text', '')}")
        sources.append((number, source_name))

    return "\n\n".join(context_parts), sources

def answer_question(question, index):
    results = search(question, index, top_k=TOP_K)
    context, sources = build_context(results)

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
Du bist PV-Guide BW, ein Assistent zur Photovoltaikpflicht
in Baden-Württemberg.

Beantworte Fragen ausschließlich anhand der bereitgestellten Quellen.

Regeln:
- Erfinde keine rechtlichen Voraussetzungen oder Ausnahmen.
- Wenn die Quellen keine ausreichende Antwort enthalten, sage das ausdrücklich.
- Wenn die Frage zu allgemein ist, bitte um die fehlenden Informationen.
- Unterscheide klar zwischen gesetzlichen Anforderungen, Ausnahmen und freiwilligen Möglichkeiten.
- Formuliere verständlich und präzise auf Deutsch.
- Belege konkrete Aussagen mit [Quelle 1], [Quelle 2] usw.
- Wenn Quellen sich widersprechen oder unklar sind, weise darauf hin.
- Halte die Antworten kompakt, aber informativ.
"""
            },
            {
                "role": "user",
                "content": f"Frage:\n{question}\n\nGefundene Quellen:\n{context}"
            }
        ],
        think=False,
        options={"num_predict": 1024}
    )

    print("\nAntwort:\n")
    print(response.message.content or "[Keine Antwort erhalten]")

    print("\nVerwendete Quellen:")
    for number, source_name in sources:
        print(f"[Quelle {number}] {source_name}")

def main():
    index = load_index()
    print(f"PV-Guide BW gestartet ({len(index)} Textabschnitte)")
    print("Stelle eine Frage oder gib 'exit' ein.")

    while True:
        question = input("\nDeine Frage: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if question:
            answer_question(question, index)


if __name__ == "__main__":
    main()
