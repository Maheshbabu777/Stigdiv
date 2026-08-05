from __future__ import annotations

import logging
from typing import Any

import requests

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FirecrawlClient:
    """Optional client for Firecrawl web scraping and search API.
    
    Used strictly on-demand (when configured and needed) to crawl company news,
    investor data, or verify unlisted/private company details.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.firecrawl_api_key
        self.base_url = "https://api.firecrawl.dev/v1"

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def scrape_url(self, url: str) -> dict[str, Any] | None:
        """Scrape markdown content from a specific web URL."""
        if not self.is_available:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"url": url, "formats": ["markdown"]}
            resp = requests.post(f"{self.base_url}/scrape", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                markdown = data.get("data", {}).get("markdown") or data.get("markdown", "")
                return {"url": url, "markdown": markdown, "metadata": data.get("data", {}).get("metadata", {})}
            logger.debug("Firecrawl scrape error %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.debug("Firecrawl scrape exception for %s: %s", url, exc)
        return None

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Search the web and extract content for financial / company queries."""
        if not self.is_available:
            return []
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "query": query,
                "limit": limit,
                "scrapeOptions": {"formats": ["markdown"]},
            }
            resp = requests.post(f"{self.base_url}/search", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                items: list[dict[str, Any]] = []
                for item in results:
                    items.append({
                        "title": item.get("title") or item.get("metadata", {}).get("title", ""),
                        "url": item.get("url") or "",
                        "markdown": item.get("markdown", "")[:1000],
                        "description": item.get("description", ""),
                    })
                return items
            logger.debug("Firecrawl search error %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.debug("Firecrawl search exception for %s: %s", query, exc)
        return []


def get_firecrawl() -> FirecrawlClient:
    return FirecrawlClient()
