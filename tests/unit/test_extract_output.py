"""Regression tests for extract_output's message-type filtering.

The bug being pinned: extract_output used to return any message with
non-empty string content when walking the trace in reverse. The
SelfVerificationMiddleware appends a HumanMessage like
"SELF-CHECK FAILED: ... Revise your report." to trigger a revision
pass. When the agent loop terminated before producing a fresh
text-bearing AIMessage (e.g. revision attempt was tool-call only,
or step cap fired), extract_output happily returned the SELF-CHECK
HumanMessage as the agent's "final output". The graders then scored
the harness's own injection, dragging prompt and harness scores down
on phantom signal and misleading the harness optimizer into making
remedies for problems that did not exist.

extract_output must only return AIMessage content.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.deep_agent import extract_output


class TestExtractOutputMessageTypeFilter:
    def test_returns_last_aimessage_when_followed_by_self_check_human(self):
        """The original bug scenario: AIMessage report followed by an
        injected SELF-CHECK HumanMessage. extract_output must skip the
        HumanMessage and return the actual AIMessage report."""
        report = "## Quantum Computing Advances\n\nIBM announced..."
        result = {
            "messages": [
                HumanMessage(content="What are the latest advances in quantum computing?"),
                AIMessage(content=report),
                HumanMessage(
                    content=(
                        "SELF-CHECK FAILED: The output is incomplete and appears "
                        "truncated mid-list. Revise your report."
                    )
                ),
            ]
        }
        assert extract_output(result) == report

    def test_skips_tool_message_with_text_content(self):
        """ToolMessages also have string content but are not the agent's output."""
        report = "Final research report goes here."
        result = {
            "messages": [
                HumanMessage(content="task"),
                AIMessage(content=report),
                ToolMessage(content="search results: ...", tool_call_id="call_1"),
            ]
        }
        assert extract_output(result) == report

    def test_skips_aimessage_with_only_tool_calls(self):
        """An AIMessage with empty content but tool_calls should be skipped."""
        report = "Earlier substantive answer."
        result = {
            "messages": [
                HumanMessage(content="task"),
                AIMessage(content=report),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search", "args": {"q": "x"}, "id": "call_1"}
                    ],
                ),
                ToolMessage(content="result", tool_call_id="call_1"),
            ]
        }
        assert extract_output(result) == report

    def test_returns_most_recent_aimessage_when_multiple(self):
        """Multiple AIMessages: the most recent text-bearing one wins."""
        result = {
            "messages": [
                HumanMessage(content="task"),
                AIMessage(content="first attempt — wrong"),
                HumanMessage(content="SELF-CHECK FAILED: try again. Revise your report."),
                AIMessage(content="revised final answer"),
            ]
        }
        assert extract_output(result) == "revised final answer"

    def test_prefers_explicit_output_key(self):
        """If the result dict has an 'output' string, use it directly."""
        result = {
            "output": "explicit output string",
            "messages": [AIMessage(content="something else")],
        }
        assert extract_output(result) == "explicit output string"

    def test_falls_back_when_no_aimessage_present(self):
        """When there is no AIMessage at all, fall back to str(result)
        rather than returning the HumanMessage's content directly. The
        fallback dict-repr is obvious garbage to the grader (will score
        zero), but critically it is NOT the clean SELF-CHECK string the
        old code path would have returned as if it were the agent's
        own answer."""
        injected = "SELF-CHECK FAILED: ... Revise your report."
        result = {
            "messages": [
                HumanMessage(content="task"),
                HumanMessage(content=injected),
            ]
        }
        out = extract_output(result)
        # The bug was returning exactly the injected content. The fallback
        # must NOT do that.
        assert out != injected
        assert out == str(result)

    def test_empty_messages_falls_back_to_str(self):
        result: dict = {"messages": []}
        assert extract_output(result) == str(result)
