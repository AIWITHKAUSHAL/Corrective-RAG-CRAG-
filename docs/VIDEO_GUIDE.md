# Video explanation and recording guide

Suggested length: 8–10 minutes. Record your screen and your own explanation. Demonstrate the code and output together; the assignment awards 10 marks for the video.

Before recording, start the app, open the architecture explorer, and run the test suite. For a live demonstration, add your EURI key locally and verify a successful Live EURI run. Keep `.env`, terminal environment dumps, and account credentials out of the recording. Do not describe an offline trace as a live model result.

## 0:00–0:45 — Problem and project

Show the playground.

“This project implements a simplified Corrective RAG agent. Standard retrieval can find irrelevant documents. My workflow grades evidence before answering. When the evidence is poor, it rewrites the search query and tries a different retrieval strategy. I use LangGraph to make state, nodes, conditional edges, and correction loops explicit.”

Explain that the live provider is EURI, using `gemini-3.5-flash-lite` through the OpenAI Python client. The offline mode is a deterministic teaching substitute.

## 0:45–1:40 — Architecture and state

Open `/architecture`. Select **Corrected retrieval**, then advance one step at a time. Click **CRAGState** to show the actual state definition.

Explain `question` versus `query`: the original intent stays unchanged; only retrieval wording changes. Point out `documents`, `grades`, `relevant_documents`, `quality`, `retries`, and `trace`. Explain why `operator.add` appends events while other fields are replaced. Explain that the visualizer replays real offline graph executions.

## 1:40–2:30 — Retrieve and grade nodes

Open `crag/graph.py` and `crag/retrieval.py`.

“The first node scores local documents by keyword overlap. The grade node then checks each document against the original question. In live mode, the EURI model returns validated JSON with a relevant/irrelevant boolean and a reason for each document. Invalid or missing grades fail the run. Rejected evidence never enters the generation context.”

Show the local JSON corpus and unrelated distractor documents. Mention that each JSON entry is a retrieval unit and that this project uses lexical retrieval rather than embeddings.

## 2:30–3:20 — The first conditional decision

Show `after_grading`.

“Quality is accepted documents divided by retrieved documents. No documents means zero. A nonempty accepted set at or above the threshold goes directly to generation. Otherwise, while retries remain, the conditional edge routes to rewriting. At the retry limit the graph terminates correction.”

Explain that this score controls routing; it is not a confidence score or a truth guarantee.

## 3:20–4:30 — Rewrite, retrieve again, and loops

Show `rewrite_query`, `after_rewrite`, and `retrieve_again`.

“The rewrite node asks for more precise search terms while preserving the original question. It also increments the retry counter. The first correction uses BM25 on the rewritten query. The new batch returns to grading, which creates the loop. Two corrections are allowed by default.”

Point to the corresponding `add_conditional_edges` and fixed `add_edge` calls. Explain `START`, `END`, and `.compile()`.

## 4:30–5:15 — Optional web search

Click the web node in the explorer and show `crag/search.py`.

“If corrected local retrieval is still poor, the second correction can use Tavily. It requires live mode, an enabled option, and a configured key. Results retain source URLs and must pass the same grader. If web search fails, it creates no evidence. Without a web provider, the agent can still correct locally.”

If you have no Tavily key, explain the branch and show its mock-provider test. Do not claim a real web search happened.

## 5:15–6:15 — Final generation and EURI integration

Show `EuriModel` and `generate_answer`.

“The OpenAI client points at the EURI base URL and uses the requested model identifier. Credentials come from the server environment. The answer prompt includes only relevant evidence and asks for document-ID citations. Citation IDs are validated. When there is no supporting evidence, the graph returns an insufficient-evidence message without asking the LLM to invent an answer.”

Explain the `partial_evidence` outcome if some evidence remains but the relevance threshold was not met by the retry limit.

## 6:15–8:00 — Demonstrate outputs

Run these questions and inspect the Answer, Evidence, and Execution trace tabs:

1. **Direct path:** `What is corrective retrieval augmented generation?`
2. **Correction path:** `My bot finds junk`
3. **No evidence:** `Who won the 2026 intergalactic chess championship?`

In offline mode, these are repeatable examples. Show the initial empty retrieval and rewritten query for example 2. Show the bounded retry count for example 3.

Then select **Live EURI** and run a real question. Explain that the model may make different relevance decisions. Export the live JSON trace to document the actual result. If live access is unavailable, disclose it instead of presenting the offline result as live.

## 8:00–9:00 — Validation and conclusion

Run `uv run pytest -q`. Explain what the behavioral tests verify: both routes, correction loops, alternate retrieval, retry bounds, malformed grading, optional web failure, and rejected-context filtering. Show the README and startup commands.

“This is a teaching implementation. Lexical retrieval and a small corpus make behavior easy to inspect. An LLM grader can still make mistakes, and citation presence does not establish factual correctness. The next extension could be semantic retrieval or a stronger coverage check.”

Finish by showing the GitHub repository and the location of the standalone visualizer. Add the real repository link to your YouTube description.

## Suggested YouTube title

**Corrective RAG with LangGraph | EURI Gemini 3.5 Flash Lite | Full Code & Live Demo**

Use “Live Demo” only if the recording actually includes a successful live EURI request.

## Description template

```text
In this walkthrough I explain my simplified Corrective RAG agent built with LangGraph.
It retrieves and grades documents, rewrites poor queries, retries with BM25,
optionally searches the web, and generates answers with document citations.

GitHub repository: [paste your published repository link]
Architecture explorer: [paste your enabled GitHub Pages link, if configured]

Chapters:
00:00 Problem and project
00:45 Architecture and shared state
01:40 Retrieval and document grading
02:30 Conditional decisions
03:20 Query rewriting and correction loops
04:30 Optional web search
05:15 EURI generation and citations
06:15 Demonstrations
08:00 Tests and limitations
```

Adjust timestamps to the final recording.
