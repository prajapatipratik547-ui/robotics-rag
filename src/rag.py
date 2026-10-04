"""The full pipeline in one call: ask(question) = retrieve -> generate.

CLI:
  python src/rag.py "What is the scan range of the RPLiDAR A1M8?"
  python src/rag.py                      # interactive: keep asking questions
  python src/rag.py --show-context "..." # also print the retrieved chunks
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.generate import Answer, generate
from src.retrieve import dense_search


def ask(question: str, k: int = config.TOP_K) -> Answer:
    hits = dense_search(question, k)
    return generate(question, hits)


def print_answer(answer: Answer, show_context: bool = False) -> None:
    print(f"\n{answer.text}\n")
    if answer.cited:
        print("Sources:")
        for n, hit in answer.cited:
            print(f"  [{n}] {hit.citation}")
    else:
        print("Sources: none cited")
    if show_context:
        print("\nRetrieved chunks (dense):")
        for i, hit in enumerate(answer.hits, start=1):
            preview = " ".join(hit.text.split())[:160]
            print(f"  [{i}] score={hit.score:.3f}  {hit.chunk_id}\n      {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the robotics docs a question.")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    if args.question:
        print_answer(ask(" ".join(args.question), args.k), args.show_context)
        return

    print("Ask a question about the robotics docs (empty line or Ctrl+C to quit).")
    try:
        while question := input("\n> ").strip():
            print_answer(ask(question, args.k), args.show_context)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
