"""Two transparent lexical strategies; no embedding model download is needed."""

import json
import math
import re
from collections import Counter
from pathlib import Path

from crag.config import ROOT
from crag.state import Document

STOP = set(
    "a an the is are of to in on for and or with how what why does do it this that my can when after".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class LocalRetriever:
    def __init__(self, path: Path = ROOT / "data/knowledge_base.json"):
        self.documents = json.loads(path.read_text(encoding="utf-8"))
        self.terms = [Counter(tokens(d["title"] + " " + d["text"])) for d in self.documents]
        self.average_length = sum(sum(t.values()) for t in self.terms) / max(len(self.terms), 1)

    def retrieve(self, query: str, top_k: int, strategy: str = "overlap") -> list[Document]:
        query_terms = set(tokens(query))
        ranked = []
        for document, counts in zip(self.documents, self.terms):
            if strategy == "overlap":
                score = len(query_terms & counts.keys()) / max(len(query_terms), 1)
            else:
                score = 0.0
                for term in query_terms:
                    frequency = counts[term]
                    df = sum(term in c for c in self.terms)
                    idf = math.log(1 + (len(self.terms) - df + 0.5) / (df + 0.5))
                    norm = 1.5 * (0.25 + 0.75 * sum(counts.values()) / max(self.average_length, 1))
                    score += idf * frequency * 2.5 / (frequency + norm)
            if score > 0:
                ranked.append({**document, "score": round(score, 6)})
        return sorted(ranked, key=lambda d: (-d["score"], d["id"]))[:top_k]
