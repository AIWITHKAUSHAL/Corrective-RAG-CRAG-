"""Six explicit LangGraph nodes, two conditional decisions, one bounded loop."""

from langgraph.graph import END, START, StateGraph

from crag.llm import LanguageModel
from crag.retrieval import LocalRetriever
from crag.state import CRAGState, RunRequest


def initial_state(request: RunRequest) -> CRAGState:
    return {
        "question": request.question,
        "query": request.question,
        "documents": [],
        "grades": [],
        "relevant_documents": [],
        "quality": 0.0,
        "retries": 0,
        "max_retries": request.max_retries,
        "top_k": request.top_k,
        "relevance_threshold": request.relevance_threshold,
        "use_web": request.use_web,
        "web_attempted": False,
        "answer": "",
        "outcome": "",
        "trace": [],
    }


def after_grading(state: CRAGState) -> str:
    if state["relevant_documents"] and state["quality"] >= state["relevance_threshold"]:
        return "generate_answer"
    if state["retries"] >= state["max_retries"]:
        return "generate_answer"
    return "rewrite_query"


def build_graph(model: LanguageModel, retriever=None, search=None):
    retriever = retriever or LocalRetriever()

    def event(node, state, detail, **extra):
        return [
            {
                "node": node,
                "attempt": state["retries"],
                "query": state["query"],
                "detail": detail,
                **extra,
            }
        ]

    def retrieve(state: CRAGState):
        docs = retriever.retrieve(state["query"], state["top_k"], strategy="overlap")
        return {
            "documents": docs,
            "trace": event(
                "retrieve",
                state,
                f"Initial keyword retrieval found {len(docs)} documents.",
                documents=docs,
                strategy="keyword overlap",
            ),
        }

    def grade_documents(state: CRAGState):
        docs = state["documents"]
        grades = model.grade(state["question"], docs) if docs else []
        accepted = {g["id"] for g in grades if g["relevant"]}
        relevant = [d for d in docs if d["id"] in accepted]
        quality = len(relevant) / len(docs) if docs else 0.0
        updated = {**state, "relevant_documents": relevant, "quality": quality}
        decision = after_grading(updated)
        return {
            "grades": grades,
            "relevant_documents": relevant,
            "quality": quality,
            "trace": event(
                "grade_documents",
                state,
                f"Accepted {len(relevant)}/{len(docs)}; next: {decision}.",
                grades=grades,
                quality=quality,
                decision=decision,
                retry_limit_reached=state["retries"] >= state["max_retries"],
            ),
        }

    def rewrite_query(state: CRAGState):
        query = model.rewrite(state["question"], state["query"], state["grades"])
        return {
            "query": query,
            "retries": state["retries"] + 1,
            "trace": event(
                "rewrite_query",
                state,
                "Preserved the question; refined the search query.",
                rewritten_query=query,
                next_attempt=state["retries"] + 1,
            ),
        }

    def after_rewrite(state: CRAGState) -> str:
        # Always try BM25 once before spending an optional web-search request.
        if (
            state["use_web"]
            and search is not None
            and state["retries"] >= 2
            and not state["web_attempted"]
        ):
            return "web_search"
        return "retrieve_again"

    def retrieve_again(state: CRAGState):
        docs = retriever.retrieve(state["query"], state["top_k"], strategy="bm25")
        return {
            "documents": docs,
            "trace": event(
                "retrieve_again",
                state,
                f"BM25 retrieval found {len(docs)} documents.",
                documents=docs,
                strategy="BM25",
            ),
        }

    def web_search(state: CRAGState):
        try:
            docs = search.search(state["query"], state["top_k"])
            detail = f"External search returned {len(docs)} snippets; grading is required."
        except Exception:
            docs = []
            detail = "Web search failed. No search evidence was accepted; continuing with an empty batch."
        return {
            "documents": docs,
            "web_attempted": True,
            "trace": event("web_search", state, detail, documents=docs, strategy="Tavily"),
        }

    def generate_answer(state: CRAGState):
        docs = state["relevant_documents"]
        if not docs:
            answer = (
                "I don't have enough relevant evidence to answer this question. "
                "The available retrieval attempts did not find supporting documents. "
                "Add relevant material to data/knowledge_base.json or enable web search."
            )
            outcome = "insufficient_evidence"
        else:
            answer = model.generate(state["question"], docs)
            outcome = (
                "answered"
                if state["quality"] >= state["relevance_threshold"]
                else "partial_evidence"
            )
            if outcome == "partial_evidence":
                answer = (
                    "Evidence is limited: retrieval quality stayed below the threshold.\n\n"
                    + answer
                )
        return {
            "answer": answer,
            "outcome": outcome,
            "trace": event(
                "generate_answer",
                state,
                f"Finished: {outcome}.",
                sources=[d["id"] for d in docs],
                outcome=outcome,
            ),
        }

    builder = StateGraph(CRAGState)
    for node in (
        retrieve,
        grade_documents,
        rewrite_query,
        retrieve_again,
        web_search,
        generate_answer,
    ):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )
    builder.add_edge("retrieve_again", "grade_documents")
    builder.add_edge("web_search", "grade_documents")
    builder.add_conditional_edges(
        "rewrite_query",
        after_rewrite,
        {
            "retrieve_again": "retrieve_again",
            "web_search": "web_search",
        },
    )
    builder.add_edge("generate_answer", END)
    return builder.compile()
