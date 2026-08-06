from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote_plus
from xml.etree import ElementTree


@dataclass(slots=True)
class RetrievedSource:
    source_id: str
    title: str
    url: str
    snippet: str
    content: str = ""

    def as_context(self) -> str:
        text = self.content or self.snippet
        return f"[{self.source_id}] {self.title}\nURL: {self.url}\n{text[:4000]}"


class WebRetriever:
    """Lightweight external retrieval using DuckDuckGo HTML search.

    This intentionally keeps retrieval provider-independent. A production deployment can
    replace this class with an academic API, enterprise search service, or vector index.
    """

    def __init__(self, max_results: int = 5, timeout: int = 12):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[RetrievedSource]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError("Install requests and beautifulsoup4 for external retrieval") from exc
        response = requests.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={"User-Agent": "Multi-Agent-Reporter/1.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[RetrievedSource] = []
        for index, result in enumerate(soup.select(".result"), start=1):
            link = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if not link:
                continue
            results.append(
                RetrievedSource(
                    source_id=f"S{index}",
                    title=link.get_text(" ", strip=True),
                    url=link.get("href", ""),
                    snippet=snippet.get_text(" ", strip=True) if snippet else "",
                )
            )
            if len(results) >= self.max_results:
                break
        return self._hydrate(results)

    def _hydrate(self, sources: list[RetrievedSource]) -> list[RetrievedSource]:
        """Fetch readable page text, while retaining snippets if a page blocks access."""
        import requests
        from bs4 import BeautifulSoup

        hydrated: list[RetrievedSource] = []
        for source in sources:
            content = ""
            try:
                page = requests.get(
                    source.url,
                    headers={"User-Agent": "Multi-Agent-Reporter/1.0"},
                    timeout=self.timeout,
                )
                if page.ok and "text/html" in page.headers.get("content-type", ""):
                    soup = BeautifulSoup(page.text, "html.parser")
                    for node in soup(["script", "style", "nav", "footer"]):
                        node.decompose()
                    content = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            except requests.RequestException:
                pass
            hydrated.append(
                RetrievedSource(source.source_id, source.title, source.url, source.snippet, content)
            )
        return hydrated


class ArxivRetriever:
    """Search arXiv's public Atom API for research papers."""

    def __init__(self, max_results: int = 5, timeout: int = 12):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[RetrievedSource]:
        import requests

        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": self.max_results, "sortBy": "relevance"},
            headers={"User-Agent": "Multi-Agent-Reporter/1.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        atom = "http://www.w3.org/2005/Atom"
        sources: list[RetrievedSource] = []
        for index, entry in enumerate(root.findall(f"{{{atom}}}entry"), start=1):
            title = re.sub(r"\s+", " ", entry.findtext(f"{{{atom}}}title", "").strip())
            summary = re.sub(r"\s+", " ", entry.findtext(f"{{{atom}}}summary", "").strip())
            links = entry.findall(f"{{{atom}}}link")
            url = next((link.attrib.get("href", "") for link in links if link.attrib.get("rel", "alternate") == "alternate"), "")
            if title and url:
                sources.append(RetrievedSource(f"P{index}", title, url, summary, summary))
        return sources


def format_sources(sources: list[RetrievedSource]) -> str:
    if not sources:
        return "No external sources were retrieved. State uncertainty explicitly."
    return "\n\n".join(source.as_context() for source in sources)
