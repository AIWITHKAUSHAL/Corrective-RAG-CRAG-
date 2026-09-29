"""Optional external retrieval; returned snippets are graded before generation."""

import httpx

from crag.state import Document


class TavilySearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, top_k: int) -> list[Document]:
        response = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "max_results": top_k,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=20,
        )
        response.raise_for_status()
        documents = []
        for index, row in enumerate(response.json().get("results", [])[:top_k], 1):
            if not row.get("content") or not row.get("url", "").startswith(("https://", "http://")):
                continue
            documents.append(
                Document(
                    id=f"web-{index}",
                    title=str(row.get("title", "Web result")),
                    text=str(row["content"])[:6000],
                    source=row["url"],
                    score=float(row.get("score") or 0),
                )
            )
        return documents
