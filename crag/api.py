"""Local web application. Run: uvicorn crag.api:app --reload."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from crag.config import ROOT, Settings
from crag.llm import ProviderError
from crag.retrieval import LocalRetriever
from crag.service import run
from crag.state import RunRequest

app = FastAPI(title="Corrective RAG Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/architecture", include_in_schema=False)
def architecture():
    return FileResponse(ROOT / "docs/architecture_visualizer.html")


@app.get("/api/config")
def config():
    settings = Settings.from_env()
    return {
        "live_ready": bool(settings.api_key),
        "web_ready": bool(settings.tavily_key),
        "model": settings.model,
        "document_count": len(LocalRetriever().documents),
    }


@app.get("/api/documents")
def documents():
    return LocalRetriever().documents


@app.post("/api/run")
def run_graph(request: RunRequest):
    try:
        return run(request)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The workflow failed. Check the server configuration and local corpus.",
        ) from exc
