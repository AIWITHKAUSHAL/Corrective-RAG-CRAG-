import json

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from crag.api import app
from crag.config import Settings
from crag.llm import EuriModel, ProviderError
from crag.retrieval import LocalRetriever
from crag.search import TavilySearch

client = TestClient(app)


def fake_model(content):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gemini-3.5-flash-lite",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model = EuriModel(Settings(api_key="test-only"))
    model.client.close()
    model.client = OpenAI(
        api_key="test-only",
        base_url="https://api.euron.one/api/v1/euri",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return model, seen


def test_exact_euri_model_endpoint_and_payload():
    model, seen = fake_model(
        '{"grades":[{"id":"euri","relevant":true,"reason":"Direct evidence"}]}'
    )
    docs = [d for d in LocalRetriever().documents if d["id"] == "euri"]
    assert model.grade("Which model?", docs)[0]["relevant"] is True
    assert str(seen[0].url) == "https://api.euron.one/api/v1/euri/chat/completions"
    assert json.loads(seen[0].content)["model"] == "gemini-3.5-flash-lite"
    model.close()


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"grades":[]}',
        '{"grades":[{"id":"euri","relevant":"true","reason":"bad boolean"}]}',
        '{"grades":[{"id":"unknown","relevant":true,"reason":"wrong id"}]}',
        '{"grades":[{"id":"euri","relevant":true,"reason":"one"},{"id":"euri","relevant":true,"reason":"two"}]}',
    ],
)
def test_invalid_grader_output_is_never_accepted(content):
    model, _ = fake_model(content)
    with pytest.raises(ProviderError, match="invalid grading"):
        model.grade("Question", [{"id": "euri"}])
    model.close()


@pytest.mark.parametrize("content", ["No citation.", "Invented citation [unknown]."])
def test_generated_answer_must_cite_known_evidence(content):
    model, _ = fake_model(content)
    with pytest.raises(ProviderError, match="citations|source"):
        model.generate("Question", [{"id": "euri"}])
    model.close()


def test_provider_failure_does_not_expose_key():
    model = EuriModel(Settings(api_key="SECRET-TEST-KEY"))
    model.client.close()
    model.client = OpenAI(
        api_key="SECRET-TEST-KEY",
        max_retries=0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(401, json={"error": "SECRET-TEST-KEY"})
            )
        ),
    )
    with pytest.raises(ProviderError) as error:
        model.generate("Question", [])
    assert "SECRET-TEST-KEY" not in str(error.value)
    model.close()


def test_tavily_snippets_have_safe_source_urls(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer test"
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "results": [
                    {"content": "Evidence", "url": "https://example.org", "score": 0.8},
                    {"content": "Unsafe link", "url": "javascript:alert(1)"},
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    docs = TavilySearch("test").search("question", 3)
    assert len(docs) == 1
    assert docs[0]["id"] == "web-1"


def test_api_demo_and_document_routes():
    result = client.post("/api/run", json={"question": "My bot finds junk"})
    assert result.status_code == 200
    assert result.json()["retries"] > 0
    assert client.get("/").status_code == 200
    assert client.get("/api/documents").status_code == 200
    assert "api_key" not in client.get("/api/config").json()


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "  "},
        {"question": "Question", "top_k": 0},
        {"question": "Question", "max_retries": 40},
        {"question": "Question", "relevance_threshold": 0},
    ],
)
def test_api_rejects_invalid_input(payload):
    assert client.post("/api/run", json=payload).status_code == 422


def test_missing_live_key_returns_clear_error(monkeypatch):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: Settings()))
    response = client.post("/api/run", json={"question": "What is CRAG?", "mode": "live"})
    assert response.status_code == 502
    assert "EURI_API_KEY" in response.json()["detail"]
