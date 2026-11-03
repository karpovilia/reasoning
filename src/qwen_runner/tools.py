from __future__ import annotations

import html
import json
import os
import re
from typing import Any
from urllib.parse import quote

import httpx



def _infer_backend(base_url: str) -> str:
    explicit = os.getenv("WIKI_BACKEND")
    if explicit:
        return explicit.strip().lower()
    return "wikipedia" if "wikipedia.org" in base_url else "kiwix"


class WikipediaToolExecutor:
    DEFAULT_USER_AGENT = (
        "qwen-runner/0.1 "
        "(https://github.com/anthropics/reasoning_trajectories; benchmark evaluation)"
    )

    # ---- configuration via env (keeps online mode working, adds offline kiwix) ----
    # WIKI_BASE_URL : endpoint root. Default https://en.wikipedia.org (live).
    #                 For offline kiwix-serve set e.g. http://localhost:8080
    # WIKI_BACKEND  : 'wikipedia' (MediaWiki action/REST API) or 'kiwix'
    #                 (kiwix-serve). Inferred from WIKI_BASE_URL when unset.
    # KIWIX_BOOK    : kiwix-serve book name for URL paths. Auto-discovered from
    #                 /catalog/v2/entries when unset.

    _TAG_RE = re.compile(r"<[^>]+>")
    _WS_RE = re.compile(r"\s+")
    _EXTRACT_CHARS = 2000
    # kiwix-serve /search result item: <li><a href="/content/<id>/<path>">Title</a> ... <cite>snippet</cite></li>
    # (footer pagination anchors point at /search?... and are excluded by the href prefix.)
    _RESULT_RE = re.compile(
        r'<li>\s*<a\s+href="((?:/content/|[^"]*?/viewer#)[^"]+)"[^>]*>(.*?)</a>(.*?)</li>',
        re.S | re.I,
    )

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.base_url = os.getenv("WIKI_BASE_URL", "https://en.wikipedia.org").rstrip("/")
        self.backend = _infer_backend(self.base_url)
        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers={
                "User-Agent": self.DEFAULT_USER_AGENT,
                "Accept": "application/json",
                "Api-User-Agent": self.DEFAULT_USER_AGENT,
            },
            timeout=30.0,
        )
        self._kiwix_book: str | None = os.getenv("KIWIX_BOOK") or None
        # title -> content path cache (kiwix), so wikipedia_page can fetch the exact path.
        self._path_cache: dict[str, str] = {}

    # ------------------------------------------------------------------ tools
    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_search",
                    "description": "Search English Wikipedia articles relevant to the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_page",
                    "description": "Fetch a Wikipedia page summary by exact article title.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Exact Wikipedia article title.",
                            }
                        },
                        "required": ["title"],
                    },
                },
            },
        ]

    def execute_tool_call(self, tool_name: str, arguments_json: str) -> dict[str, Any]:
        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            arguments = {}
        if tool_name == "wikipedia_search":
            query = arguments.get("query", "")
            if not query:
                return {"query": "", "results": [], "error": "missing query argument"}
            return self.wikipedia_search(query)
        if tool_name == "wikipedia_page":
            title = arguments.get("title", "")
            if not title:
                return {"title": "", "extract": "", "error": "missing title argument"}
            return self.wikipedia_page(title)
        raise ValueError(f"Unsupported tool: {tool_name}")

    def wikipedia_search(self, query: str) -> dict[str, Any]:
        if self.backend == "kiwix":
            return self._kiwix_search(query)
        return self._wikipedia_search(query)

    def wikipedia_page(self, title: str) -> dict[str, Any]:
        if self.backend == "kiwix":
            return self._kiwix_page(title)
        return self._wikipedia_page(title)

    # ------------------------------------------------------- live MediaWiki API
    def _wikipedia_search(self, query: str) -> dict[str, Any]:
        try:
            response = self.client.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "format": "json",
                    "srsearch": query,
                    "srlimit": 5,
                    "utf8": 1,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "query": query,
                "results": [],
                "error": f"wikipedia_search failed with HTTP {exc.response.status_code}",
            }
        except httpx.HTTPError as exc:
            return {"query": query, "results": [], "error": f"wikipedia_search transport error: {exc}"}
        payload = response.json()
        results = [
            {
                "title": item["title"],
                "snippet": item.get("snippet", ""),
            }
            for item in payload.get("query", {}).get("search", [])
        ]
        return {"query": query, "results": results}

    def _wikipedia_page(self, title: str) -> dict[str, Any]:
        summary_path = f"/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}"
        try:
            response = self.client.get(summary_path)
            response.raise_for_status()
            payload = response.json()
            return {
                "title": payload.get("title", title),
                "extract": payload.get("extract", ""),
                "url": payload.get("content_urls", {}).get("desktop", {}).get("page"),
            }
        except httpx.HTTPStatusError:
            return self._wikipedia_page_fallback(title)
        except httpx.HTTPError as exc:
            return {"title": title, "extract": "", "error": f"wikipedia_page transport error: {exc}"}

    def _wikipedia_page_fallback(self, title: str) -> dict[str, Any]:
        try:
            response = self.client.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts|info",
                    "inprop": "url",
                    "explaintext": 1,
                    "redirects": 1,
                    "titles": title,
                    "format": "json",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "title": title,
                "extract": "",
                "error": f"wikipedia_page failed with HTTP {exc.response.status_code}",
            }
        except httpx.HTTPError as exc:
            return {"title": title, "extract": "", "error": f"wikipedia_page transport error: {exc}"}
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return {
            "title": page.get("title", title),
            "extract": page.get("extract", ""),
            "url": page.get("fullurl"),
        }

    # ----------------------------------------------------------- offline kiwix
    # kiwix-serve (libkiwix 14) API as observed on wikipedia_en_*_nopic ZIMs:
    #   * /suggest is NOT available (404) when a single ZIM is served directly.
    #   * bare /search?pattern=<q>&pageLength=<n> full-text-searches the served
    #     book and returns an HTML result list; each <li> has an anchor
    #     href="/content/<content-id>/<path>" (no A/ namespace) plus a <cite> snippet.
    #   * the <content-id> is the ZIM filename stem, NOT the OPDS <name> metadata.
    # Strategy: parse /search HTML and reuse the result href verbatim to fetch the
    # page, so we never have to reconstruct content-id / path ourselves.
    def _strip_html(self, raw: str) -> str:
        # Drop script/style blocks then all tags; unescape entities; collapse WS.
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        text = self._TAG_RE.sub(" ", raw)
        text = html.unescape(text)
        return self._WS_RE.sub(" ", text).strip()

    def _content_book_id(self) -> str:
        """Book id used in /content/<id>/... URLs (ZIM filename stem), read from the
        OPDS catalog's text/html link. Used only for the last-resort direct fetch."""
        if self._kiwix_book:
            return self._kiwix_book
        try:
            resp = self.client.get("/catalog/v2/entries", params={"count": 1})
            resp.raise_for_status()
            m = re.search(r'href="/content/([^"/]+)"', resp.text)
            if m:
                self._kiwix_book = m.group(1)
                return self._kiwix_book
        except httpx.HTTPError:
            pass
        return self._kiwix_book or ""

    def _kiwix_search_parse(self, query: str) -> list[dict[str, str]]:
        try:
            resp = self.client.get(
                "/search", params={"pattern": query, "pageLength": 5}
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        results: list[dict[str, str]] = []
        for m in self._RESULT_RE.finditer(resp.text):
            href, anchor, rest = m.group(1), m.group(2), m.group(3)
            title = self._strip_html(anchor)
            if not title:
                continue
            cm = re.search(r"<cite[^>]*>(.*?)</cite>", rest, re.S | re.I)
            snippet = self._strip_html(cm.group(1)) if cm else ""
            results.append({"title": title, "snippet": snippet, "href": href})
            if len(results) >= 5:
                break
        return results

    def _kiwix_search(self, query: str) -> dict[str, Any]:
        hits = self._kiwix_search_parse(query)
        results = []
        for h in hits[:5]:
            self._path_cache[h["title"]] = h["href"]
            results.append({"title": h["title"], "snippet": h.get("snippet", "")})
        return {"query": query, "results": results}

    def _kiwix_fetch(self, href: str) -> str | None:
        try:
            resp = self.client.get(href)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError:
            return None

    def _kiwix_page(self, title: str) -> dict[str, Any]:
        href = self._path_cache.get(title)
        if not href:
            hits = self._kiwix_search_parse(title)
            for h in hits:
                if h["title"].strip().lower() == title.strip().lower():
                    href = h["href"]
                    break
            if not href and hits:
                href = hits[0]["href"]
            if href:
                self._path_cache[title] = href
        if not href:
            cid = self._content_book_id()
            if cid:
                href = f"/content/{cid}/{quote(title.replace(' ', '_'))}"
        if not href:
            return {"title": title, "extract": "", "error": "kiwix page not found"}
        html_text = self._kiwix_fetch(href)
        if not html_text:
            return {"title": title, "extract": "", "error": "kiwix page not found"}
        page_title = title
        tm = re.search(r"<title>(.*?)</title>", html_text, re.S | re.I) or re.search(
            r"<h1[^>]*>(.*?)</h1>", html_text, re.S | re.I
        )
        if tm:
            page_title = self._strip_html(tm.group(1)) or title
        extract = self._strip_html(html_text)[: self._EXTRACT_CHARS]
        return {"title": page_title, "extract": extract, "url": f"{self.base_url}{href}"}
