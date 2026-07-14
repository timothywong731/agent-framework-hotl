"""Shared test doubles: duck-typed workflow context and a scripted agent.

Executor ``@handler`` methods are plain async methods, so tests call them
directly with these fakes - no workflow, no LLM.
"""


class FakeCtx:
    """Duck-typed WorkflowContext capturing outbound messages/requests/outputs.

    Attributes:
        sent: Messages passed to :meth:`send_message`, in order.
        requests: Payloads passed to :meth:`request_info`, in order.
        outputs: Values passed to :meth:`yield_output`, in order.

    Example:
        >>> ctx = FakeCtx()
        >>> import asyncio; asyncio.run(ctx.send_message("hi"))
        >>> ctx.sent
        ['hi']
    """

    def __init__(self):
        self.sent = []
        self.requests = []
        self.outputs = []

    async def send_message(self, message):
        """Record a message an executor would route along graph edges."""
        self.sent.append(message)

    async def request_info(self, request_data, response_type=str):
        """Record a human-input request (the workflow-pause mechanism)."""
        self.requests.append(request_data)

    async def yield_output(self, output):
        """Record a terminal workflow output."""
        self.outputs.append(output)


class FakeAgentResult:
    """Minimal stand-in for an agent-framework run result: just ``.text``."""

    def __init__(self, text):
        self.text = text


class FakeAgent:
    """Scripted stand-in for ``agent_framework.Agent``.

    Returns queued texts one per :meth:`run` call (empty string once
    exhausted) and records every prompt for assertions.

    Attributes:
        texts: Remaining scripted responses.
        prompts: Prompts received so far.
        side_effect: Optional callable invoked with each prompt - used to
            simulate tool side effects (e.g. writing memory) during a run.

    Example:
        >>> agent = FakeAgent(["# Report"])
        >>> import asyncio; asyncio.run(agent.run("go")).text
        '# Report'
        >>> agent.prompts
        ['go']
    """

    def __init__(self, texts, side_effect=None):
        self.texts = list(texts)
        self.prompts = []
        self.side_effect = side_effect  # optional callable(prompt) run per call

    async def run(self, prompt):
        """Record the prompt, fire the side effect, pop the next scripted text."""
        self.prompts.append(prompt)
        if self.side_effect:
            self.side_effect(prompt)
        return FakeAgentResult(self.texts.pop(0) if self.texts else "")
