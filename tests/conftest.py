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

    Mirrors the real Agent's session API so executors can mint a session per
    run cycle and pass it to every ``run()`` in that cycle.

    Attributes:
        texts: Remaining scripted responses.
        prompts: Prompts received so far.
        sessions: The ``session`` passed to each :meth:`run` call, in order.
        created_sessions: Sessions handed out by :meth:`create_session`.
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
        self.sessions = []
        self.created_sessions = []
        self.side_effect = side_effect  # optional callable(prompt) run per call

    def create_session(self, *, session_id=None):
        """Hand out an opaque session sentinel, as the real Agent does."""
        session = f"session-{len(self.created_sessions) + 1}"
        self.created_sessions.append(session)
        return session

    async def run(self, prompt, *, session=None):
        """Record prompt + session, fire the side effect, pop the next text."""
        self.prompts.append(prompt)
        self.sessions.append(session)
        if self.side_effect:
            self.side_effect(prompt)
        return FakeAgentResult(self.texts.pop(0) if self.texts else "")


DRIVE_TARGETS = {
    "discovery": ("discovery", None),
    "analyze_oms-monolith": ("deep_analysis", "oms-monolith"),
    "analyze_oms-batch-recon": ("deep_analysis", "oms-batch-recon"),
    "enterprise_context": ("enterprise_context", None),
    "questionnaire": ("questionnaire", None),
}


class DriveAgent:
    """Stands in for every Agent when driving the REAL assembled graph LLM-free:
    raises one question per phase on the initial pass, one extra during
    discovery's revision, records call order. Shared by test_pipeline.py and
    test_checkpoint.py (the pause/resume cycle)."""

    def __init__(self, name, store, calls):
        self.name, self.store, self.calls = name, store, calls

    def create_session(self, *, session_id=None):
        """Mirror the real Agent's session API; PhaseExecutor mints one per cycle."""
        return f"{self.name}-session"

    async def run(self, prompt, *, session=None):
        if self.name == "final_report":
            self.calls.append((self.name, "report", prompt))
            return FakeAgentResult("FINAL-VERDICT")
        kind = "revision" if "## HUMAN ANSWERS" in prompt else "initial"
        self.calls.append((self.name, kind, prompt))
        phase, unit = DRIVE_TARGETS[self.name]
        if kind == "initial":
            self.store.update_memory(phase, unit, f"finding_{len(self.calls)}", "v")
            self.store.raise_question(phase, unit, f"Q from {self.name}?", "ctx", "default",
                                       importance="medium", impact="swings the verdict")
        elif self.name == "discovery":
            self.store.raise_question(phase, unit, "Raised during revision?", "ctx", "post-gate",
                                       importance="medium", impact="swings the verdict")
        return FakeAgentResult(f"REPORT[{self.name}][{kind}]")
