class FakeCtx:
    """Duck-typed WorkflowContext capturing outbound messages/requests/outputs."""

    def __init__(self):
        self.sent = []
        self.requests = []
        self.outputs = []

    async def send_message(self, message):
        self.sent.append(message)

    async def request_info(self, request_data, response_type=str):
        self.requests.append(request_data)

    async def yield_output(self, output):
        self.outputs.append(output)


class FakeAgentResult:
    def __init__(self, text):
        self.text = text


class FakeAgent:
    """Scripted stand-in for agent_framework.Agent: returns queued texts, records prompts."""

    def __init__(self, texts, side_effect=None):
        self.texts = list(texts)
        self.prompts = []
        self.side_effect = side_effect  # optional callable(prompt) run per call

    async def run(self, prompt):
        self.prompts.append(prompt)
        if self.side_effect:
            self.side_effect(prompt)
        return FakeAgentResult(self.texts.pop(0) if self.texts else "")
