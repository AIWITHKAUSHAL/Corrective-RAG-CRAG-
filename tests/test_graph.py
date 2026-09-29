"""Behavioral tests run the compiled LangGraph, not a hand-written substitute loop."""

import pytest

from crag.graph import build_graph, initial_state
from crag.llm import DemoModel
from crag.service import run
from crag.state import RunRequest

DOC = {
    "id": "good",
    "title": "Evidence",
    "text": "Supported information.",
    "source": "test",
    "score": 1.0,
}
BAD = {**DOC, "id": "bad"}


class ScriptedModel:
    def grade(self, question, documents):
        assert question == "Original question?"  # Rewrites must not change grading intent.
        return [
            {"id": d["id"], "relevant": d["id"] != "bad", "reason": "Test evidence"}
            for d in documents
        ]

    def rewrite(self, question, query, reasons):
        return "precise search query"

    def generate(self, question, documents):
        assert all(d["id"] != "bad" for d in documents)  # Rejected context must not leak.
        return "Supported answer. [good]"


class ScriptedRetriever:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def retrieve(self, query, top_k, strategy):
        self.calls.append((query, strategy))
        return next(self.batches)


class Search:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider failure")
        return [DOC]


def invoke(batches, search=None, **options):
    retriever = ScriptedRetriever(batches)
    request = RunRequest(question="Original question?", **options)
    result = build_graph(ScriptedModel(), retriever, search).invoke(initial_state(request))
    return result, retriever


def nodes(result):
    return [step["node"] for step in result["trace"]]


def test_relevant_initial_documents_generate_directly():
    result, _ = invoke([[DOC, BAD]])
    assert nodes(result) == ["retrieve", "grade_documents", "generate_answer"]
    assert result["quality"] == 0.5
    assert result["relevant_documents"] == [DOC]
    assert result["outcome"] == "answered"


def test_bad_retrieval_rewrites_switches_strategy_and_regrades():
    result, retriever = invoke([[BAD], [DOC]])
    assert nodes(result) == [
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve_again",
        "grade_documents",
        "generate_answer",
    ]
    assert retriever.calls == [("Original question?", "overlap"), ("precise search query", "bm25")]
    assert result["question"] == "Original question?"
    assert result["retries"] == 1


@pytest.mark.parametrize("maximum", [0, 1, 2, 3])
def test_retry_limit_terminates_empty_retrieval(maximum):
    result, _ = invoke([[]] * (maximum + 1), max_retries=maximum)
    assert result["retries"] == maximum
    assert nodes(result).count("rewrite_query") == maximum
    assert result["outcome"] == "insufficient_evidence"
    assert "don't have enough" in result["answer"]


def test_web_fallback_is_graded_before_generation():
    search = Search()
    result, _ = invoke([[BAD], [BAD]], search=search, use_web=True)
    assert search.calls == 1
    assert nodes(result)[-3:] == ["web_search", "grade_documents", "generate_answer"]
    assert result["outcome"] == "answered"


def test_failed_web_search_returns_insufficient_evidence():
    result, _ = invoke([[BAD], [BAD]], search=Search(fail=True), use_web=True)
    assert result["outcome"] == "insufficient_evidence"
    assert "failed" in result["trace"][-3]["detail"]


def test_missing_web_provider_uses_local_retry():
    result, _ = invoke([[], [], [DOC]], use_web=True)
    assert "web_search" not in nodes(result)
    assert result["outcome"] == "answered"


def test_partial_evidence_is_disclosed_at_retry_limit():
    result, _ = invoke([[DOC, BAD]], relevance_threshold=1, max_retries=0)
    assert result["outcome"] == "partial_evidence"
    assert result["answer"].startswith("Evidence is limited")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is corrective retrieval augmented generation?", "answered"),
        ("My bot finds junk", "answered"),
        ("Who won the 2026 intergalactic chess championship?", "insufficient_evidence"),
    ],
)
def test_real_corpus_demo_scenarios(question, expected):
    result = run(RunRequest(question=question))
    assert result["outcome"] == expected
    if question == "My bot finds junk":
        assert "rewrite_query" in nodes(result)


def test_empty_corpus_stops(tmp_path):
    from crag.retrieval import LocalRetriever

    path = tmp_path / "empty.json"
    path.write_text("[]")
    graph = build_graph(DemoModel(), LocalRetriever(path))
    result = graph.invoke(initial_state(RunRequest(question="Anything here?")))
    assert result["outcome"] == "insufficient_evidence"
