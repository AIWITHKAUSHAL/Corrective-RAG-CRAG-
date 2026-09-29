"""CLI: python -m crag 'What is CRAG?' --mode demo."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from crag.llm import ProviderError
from crag.service import run
from crag.state import RunRequest


def main():
    parser = argparse.ArgumentParser(description="Run the LangGraph Corrective RAG agent")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--output", type=Path, help="Save complete JSON trace")
    args = parser.parse_args()
    try:
        result = run(
            RunRequest(
                question=args.question,
                mode=args.mode,
                top_k=args.top_k,
                max_retries=args.max_retries,
                relevance_threshold=args.threshold,
                use_web=args.web,
            )
        )
    except (ProviderError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Mode: {result['mode']} | Model: {result['model']}")
    for step in result["trace"]:
        print(f"  [{step['node']}] {step['detail']}")
    print("\n" + result["answer"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nTrace saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
