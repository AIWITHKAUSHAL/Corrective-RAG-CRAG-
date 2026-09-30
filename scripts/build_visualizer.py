"""Generate a standalone explorer from actual source and executed traces.

The three demo scenarios run offline. The web scenario calls EURI and Tavily for
real, so it needs network access and both keys; without them the last saved live
recording in docs/demo_runs.json is reused.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crag.config import Settings  # noqa: E402
from crag.graph import build_graph  # noqa: E402
from crag.llm import DemoModel  # noqa: E402
from crag.service import run  # noqa: E402
from crag.state import RunRequest  # noqa: E402

SCENARIOS = {
    "direct": "What is corrective retrieval augmented generation?",
    "corrected": "My bot finds junk",
    "unsupported": "Who won the 2026 intergalactic chess championship?",
}
README_MARKERS = ("<!-- langgraph:start -->", "<!-- langgraph:end -->")
WEB_QUESTION = "At what temperature does water boil at the summit of Mount Everest?"


def web_run():
    settings = Settings.from_env()
    saved = ROOT / "docs/demo_runs.json"
    previous = json.loads(saved.read_text(encoding="utf-8")).get("web") if saved.exists() else None
    if not (settings.api_key and settings.tavily_key):
        if previous is None:
            sys.exit("The web scenario needs EURI_API_KEY and TAVILY_API_KEY for its first recording.")
        print("EURI/Tavily keys not set; reusing the saved live web recording.")
        return previous
    try:
        result = run(RunRequest(question=WEB_QUESTION, mode="live", use_web=True), settings)
    except Exception as exc:
        if previous is None:
            raise
        print(f"Live web recording failed ({exc}); reusing the saved recording.")
        return previous
    if "web_search" not in [step["node"] for step in result["trace"]]:
        sys.exit("The live run answered without web search; choose a question outside the corpus.")
    return result


def main():
    source_files = [
        "crag/graph.py",
        "crag/state.py",
        "crag/retrieval.py",
        "crag/llm.py",
        "crag/search.py",
        "crag/service.py",
        "crag/api.py",
    ]
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in source_files}
    excerpts = {}
    for filename, source in sources.items():
        for item in ast.walk(ast.parse(source)):
            if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
                excerpts[f"{filename}:{item.name}"] = ast.get_source_segment(source, item)
    runs = {key: run(RunRequest(question=question)) for key, question in SCENARIOS.items()}
    runs["web"] = web_run()
    diagram = build_graph(DemoModel()).get_graph().draw_mermaid().rstrip()
    payload = {"sources": sources, "excerpts": excerpts, "runs": runs, "mermaid": diagram}
    (ROOT / "docs/demo_runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    template = (ROOT / "scripts/visualizer_template.html").read_text(encoding="utf-8")
    # Prevent source strings from closing the data script element.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (ROOT / "docs/architecture_visualizer.html").write_text(
        template.replace("__PROJECT_DATA__", encoded),
        encoding="utf-8",
    )
    (ROOT / "docs/workflow.mmd").write_text(diagram + "\n", encoding="utf-8")
    sync_readme(diagram)
    print("Built docs/architecture_visualizer.html, demo_runs.json, workflow.mmd, and README graph")


def sync_readme(diagram):
    """Replace the README block between the langgraph markers with the compiled graph."""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = README_MARKERS
    if start not in text or end not in text:
        sys.exit(f"README.md is missing the {start} / {end} markers.")
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    block = f"{start}\n```mermaid\n{diagram}\n```\n{end}"
    readme.write_text(head + block + tail, encoding="utf-8")

if __name__ == "__main__":
    main()
