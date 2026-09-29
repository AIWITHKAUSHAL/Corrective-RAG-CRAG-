"""Shared runner used by the browser, CLI, and saved examples."""

import time

from crag.config import Settings
from crag.graph import build_graph, initial_state
from crag.llm import DemoModel, EuriModel
from crag.search import TavilySearch
from crag.state import RunRequest


def run(request: RunRequest, settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    model = EuriModel(settings) if request.mode == "live" else DemoModel()
    search = (
        TavilySearch(settings.tavily_key)
        if (request.mode == "live" and request.use_web and settings.tavily_key)
        else None
    )
    warnings = []
    if request.use_web and search is None:
        warnings.append(
            "Web search unavailable: use live mode and configure TAVILY_API_KEY. Local correction remains enabled."
        )
    start = time.perf_counter()
    try:
        state = build_graph(model, search=search).invoke(
            initial_state(request),
            {"recursion_limit": 30},
        )
    finally:
        if isinstance(model, EuriModel):
            model.close()
    return {
        **state,
        "mode": request.mode,
        "model": settings.model if request.mode == "live" else "Deterministic demo (no LLM)",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "warnings": warnings,
    }
