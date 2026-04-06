"""Web search tool with Tavily primary and DuckDuckGo fallback.

Tavily is used as the primary search engine (better quality, structured results).
When Tavily fails (rate limits, plan quotas, API errors), DuckDuckGo is used as
a free fallback that requires no API key. If both fail, returns a graceful message
telling the agent to proceed with its existing knowledge.
"""

import logging
import time
from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

from src.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 3
DEFAULT_SEARCH_DEPTH = "basic"
MAX_RETRIES = 1
RETRY_DELAY_SECONDS = 2


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception indicates an unrecoverable quota/plan limit."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("limit", "quota", "plan", "429", "432", "rate"))


class ResilientSearch(BaseTool):
    """Search tool with Tavily primary and DuckDuckGo fallback.

    Execution order:
    1. Try Tavily (better quality, structured results)
    2. If Tavily fails with a quota/plan error, fall back to DuckDuckGo
    3. If Tavily fails with a transient error, retry once, then fall back to DDG
    4. If DuckDuckGo also fails, return a graceful message (never raise)
    """

    name: str = "web_search"
    description: str = (
        "Search the web for current information on a topic. "
        "Returns search results with titles, URLs, and content snippets."
    )
    tavily_tool: TavilySearch
    ddg_tool: DuckDuckGoSearchResults
    max_retries: int = MAX_RETRIES
    retry_delay: int = RETRY_DELAY_SECONDS

    class Config:
        arbitrary_types_allowed = True

    def _search_tavily(self, query: str) -> str | None:
        """Try Tavily search. Returns result string or None on failure."""
        for attempt in range(1 + self.max_retries):
            try:
                result = self.tavily_tool._run(query)
                # Tavily may return {'error': ...} dict instead of raising
                if isinstance(result, dict) and "error" in result:
                    error_msg = str(result["error"])
                    logger.warning("Tavily returned error dict: %s", error_msg)
                    if _is_quota_error(Exception(error_msg)):
                        return None  # Fall through to DDG
                    continue
                if result and isinstance(result, str):
                    return result
                if result:
                    return str(result)
                return None
            except Exception as exc:
                if _is_quota_error(exc):
                    logger.warning("Tavily quota/rate limit hit: %s", exc)
                    return None  # Fall through to DDG immediately
                if attempt < self.max_retries:
                    logger.warning(
                        "Tavily search failed (attempt %d/%d): %s — retrying",
                        attempt + 1,
                        1 + self.max_retries,
                        exc,
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning("Tavily search failed after retries: %s", exc)
        return None

    def _search_ddg(self, query: str) -> str | None:
        """Try DuckDuckGo search. Returns result string or None on failure."""
        try:
            result = self.ddg_tool.invoke(query)
            if result:
                return result
            return None
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return None

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Run search with Tavily -> DuckDuckGo fallback chain."""
        # 1. Try Tavily (primary)
        result = self._search_tavily(query)
        if result:
            return result

        # 2. Fall back to DuckDuckGo (free, no API key needed)
        logger.info("Falling back to DuckDuckGo for: %s", query[:60])
        result = self._search_ddg(query)
        if result:
            return result

        # 3. Both failed — return graceful message
        return (
            f"Search returned no results for: {query}. "
            "Use your existing knowledge to answer this sub-question."
        )


def create_search_tool(settings: Settings, harness_config=None) -> BaseTool:
    """Create a search tool with Tavily primary and DuckDuckGo fallback.

    Tavily provides higher quality results but requires an API key and has
    plan limits. DuckDuckGo is free and unlimited but returns less structured
    results. The tool tries Tavily first and falls back to DDG on failure.

    When *harness_config* is provided, its search_* fields override the
    module-level defaults for max_results, search_depth, max_retries, and
    retry_delay.
    """
    max_results = DEFAULT_MAX_RESULTS
    search_depth = DEFAULT_SEARCH_DEPTH
    max_retries = MAX_RETRIES
    retry_delay = RETRY_DELAY_SECONDS

    if harness_config is not None:
        max_results = getattr(harness_config, "search_max_results", max_results)
        search_depth = getattr(harness_config, "search_depth", search_depth)
        max_retries = getattr(harness_config, "search_max_retries", max_retries)
        retry_delay = getattr(harness_config, "search_retry_delay", retry_delay)

    tavily = TavilySearch(
        max_results=max_results,
        search_depth=search_depth,
        tavily_api_key=settings.tavily_api_key,
    )
    ddg = DuckDuckGoSearchResults(num_results=max_results)

    return ResilientSearch(
        tavily_tool=tavily,
        ddg_tool=ddg,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
