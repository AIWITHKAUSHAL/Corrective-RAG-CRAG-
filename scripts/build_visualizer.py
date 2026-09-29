"""Generate a standalone explorer from actual source and executed offline traces."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crag.graph import build_graph  # noqa: E402
from crag.llm import DemoModel  # noqa: E402
from crag.service import run  # noqa: E402
from crag.state import RunRequest  # noqa: E402

SCENARIOS = {
    "direct": "What is corrective retrieval augmented generation?",
    "corrected": "My bot finds junk",
    "unsupported": "Who won the 2026 intergalactic chess championship?",
}


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
    payload = {"sources": sources, "excerpts": excerpts, "runs": runs}
    (ROOT / "docs/demo_runs.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    template = (ROOT / "scripts/visualizer_template.html").read_text(encoding="utf-8")
    # Prevent source strings from closing the data script element.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (ROOT / "docs/architecture_visualizer.html").write_text(
        template.replace("__PROJECT_DATA__", encoded),
        encoding="utf-8",
    )
    diagram = build_graph(DemoModel()).get_graph().draw_mermaid()
    (ROOT / "docs/workflow.mmd").write_text(diagram.rstrip() + "\n", encoding="utf-8")
    print("Built docs/architecture_visualizer.html, demo_runs.json, and workflow.mmd")


if __name__ == "__main__":
    main()
