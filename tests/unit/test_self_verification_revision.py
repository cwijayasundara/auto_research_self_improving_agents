"""Regression test: SelfVerificationMiddleware actually triggers a revision pass.

The bug being pinned: the middleware used to append a "SELF-CHECK FAILED ...
Revise your report" HumanMessage to state from `after_model`, then return.
But LangChain's routing function (`_make_model_to_tools_edge` in
`langchain.agents.factory`) walks the message list backwards to find the LAST
AIMessage, checks ITS tool_calls, and exits to END if none. The injected
HumanMessage is silently ignored. As a result, max_retries was dead config
and the model was called exactly ONCE per task — never given a chance to
revise the output.

Fix: decorate `after_model` with `@hook_config(can_jump_to=["model"])` and
return `{"jump_to": "model"}` alongside the new message. That wires a
conditional edge from the after_model node back to the model node and tells
the routing to take it.

This test uses a real langchain agent + the real middleware + a fake LLM
that always returns short content. It counts model invocations to prove the
revision pass runs.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from evoagent.harness.middleware import SelfVerificationMiddleware


class _CountingFakeLLM(BaseChatModel):
    """A fake chat model that counts invocations and always returns short text.

    Uses a list-as-cell for the counter because BaseChatModel is a pydantic
    model and won't allow plain class/instance attributes for mutable state.
    """

    @property
    def _llm_type(self) -> str:
        return "counting-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Use module-level counter passed in via class attr swap below
        _CALLS[0] += 1
        msg = AIMessage(content=f"too short response #{_CALLS[0]}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


_CALLS = [0]


class TestSelfVerificationRevisionTriggers:
    def setup_method(self) -> None:
        _CALLS[0] = 0

    def test_revision_pass_runs_max_retries_times(self):
        """With min_length far above what the fake LLM produces and
        max_retries=2, the model should be called 3 times total
        (initial attempt + 2 revision retries)."""
        llm = _CountingFakeLLM()
        middleware = SelfVerificationMiddleware(
            required_sections=[],          # disable section check, focus on length
            min_length=500,
            max_retries=2,
            verify_against_task=False,
        )

        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt="be brief",
            middleware=[middleware],
        )

        result = agent.invoke({"messages": [HumanMessage(content="What is X?")]})

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        revision_messages = [
            m for m in result["messages"]
            if isinstance(m, HumanMessage) and "SELF-CHECK FAILED" in (m.content or "")
        ]

        assert _CALLS[0] == 3, (
            f"Model should be called 3 times (initial + 2 retries), got {_CALLS[0]}. "
            f"If this is 1, the after_model hook is not jumping back to the model "
            f"node — check that @hook_config(can_jump_to=['model']) decorator is "
            f"present and the return dict includes 'jump_to': 'model'."
        )
        assert len(ai_messages) == 3, (
            f"Expected 3 AIMessages in final state, got {len(ai_messages)}"
        )
        assert len(revision_messages) == 2, (
            f"Expected 2 SELF-CHECK revision injections, got {len(revision_messages)}"
        )

    def test_no_revision_when_output_passes_check(self):
        """When the output passes the self-check, the model should be
        called exactly once and no revision should be injected."""
        # Build an LLM that produces a response that satisfies BOTH the default
        # min_length AND the default required_sections (summary/finding/source).
        # Note: passing required_sections=[] to the middleware would be silently
        # coerced back to defaults because of the `x or default` pattern in
        # __init__ — that's a separate latent bug worth tracking, but here we
        # just work around it.
        long_text = (
            "## Summary\n"
            "This is a thorough finding about the topic.\n"
            "Source: example.com\n"
        ) + "padding " * 100

        class GoodLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "good"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                _CALLS[0] += 1
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=long_text))]
                )

        llm = GoodLLM()
        middleware = SelfVerificationMiddleware(
            required_sections=[],
            min_length=500,
            max_retries=2,
            verify_against_task=False,
        )

        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt="be thorough",
            middleware=[middleware],
        )

        result = agent.invoke({"messages": [HumanMessage(content="What is X?")]})

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        revision_messages = [
            m for m in result["messages"]
            if isinstance(m, HumanMessage) and "SELF-CHECK FAILED" in (m.content or "")
        ]

        assert _CALLS[0] == 1
        assert len(ai_messages) == 1
        assert len(revision_messages) == 0

    def test_max_retries_zero_means_no_revision(self):
        """max_retries=0 disables the revision pass — model called exactly once
        even when output fails the check."""
        llm = _CountingFakeLLM()
        middleware = SelfVerificationMiddleware(
            required_sections=[],
            min_length=500,
            max_retries=0,
            verify_against_task=False,
        )

        agent = create_agent(
            model=llm,
            tools=[],
            system_prompt="be brief",
            middleware=[middleware],
        )

        agent.invoke({"messages": [HumanMessage(content="What is X?")]})
        assert _CALLS[0] == 1
