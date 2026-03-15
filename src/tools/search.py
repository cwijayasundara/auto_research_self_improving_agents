"""Tavily web search tool factory."""

import logging

from langchain_tavily import TavilySearch

from src.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_DEPTH = "advanced"


def create_search_tool(settings: Settings) -> TavilySearch:
    """Create a TavilySearch tool configured for research."""
    return TavilySearch(
        max_results=DEFAULT_MAX_RESULTS,
        search_depth=DEFAULT_SEARCH_DEPTH,
        tavily_api_key=settings.tavily_api_key,
    )
