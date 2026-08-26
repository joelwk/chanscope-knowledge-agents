from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider
import pytest


def _make_agent():
    # Avoid KnowledgeAgent.__init__ (requires API keys / client init).
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    agent._unsupported_params_by_model = {}
    agent._models_without_temperature = set()
    agent.use_deterministic_chunking = False
    agent.chunk_size = 1000
    agent.chunk_overlap = 200
    agent.max_retries = 3
    return agent


def test_prepare_model_params_omits_temperature_for_gpt5():
    agent = _make_agent()
    params = agent._prepare_model_params(
        ModelProvider.OPENAI,
        {"model": "gpt-5.2-2025-12-11", "messages": []},
    )
    assert "temperature" not in params
    assert "presence_penalty" not in params
    assert "frequency_penalty" not in params


def test_prepare_model_params_keeps_penalties_for_gpt4():
    agent = _make_agent()
    params = agent._prepare_model_params(
        ModelProvider.OPENAI,
        {"model": "gpt-4o", "messages": []},
    )
    assert "temperature" in params
    assert "presence_penalty" in params
    assert "frequency_penalty" in params


def test_extract_unsupported_param_name_from_body():
    class Dummy(Exception):
        def __init__(self):
            self.body = {
                "error": {
                    "message": "Unsupported parameter: 'presence_penalty' is not supported with this model.",
                    "type": "invalid_request_error",
                    "param": "presence_penalty",
                    "code": "unsupported_parameter",
                }
            }
            super().__init__("Error code: 400 - ...")

    assert KnowledgeAgent._extract_unsupported_param_name(Dummy()) == "presence_penalty"


def test_extract_unsupported_param_name_from_message():
    class Dummy(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'presence_penalty' is not supported with this model.\", 'param': 'presence_penalty', 'code': 'unsupported_parameter'}}"
            )

    assert KnowledgeAgent._extract_unsupported_param_name(Dummy()) == "presence_penalty"


def test_extract_unsupported_temperature_value_from_log_error():
    class Dummy(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 400 - {'error': {'message': \"Unsupported value: 'temperature' does not support 0.3 with this model. Only the default (1) value is supported.\", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'unsupported_value'}}"
            )

    assert KnowledgeAgent._extract_unsupported_param_name(Dummy()) == "temperature"
    assert KnowledgeAgent._is_non_retryable_api_error(Dummy()) is True


def test_cached_unsupported_params_are_removed():
    agent = _make_agent()
    agent._mark_unsupported_params_for_model("gpt-4o", ["presence_penalty"])

    params = agent._prepare_model_params(
        ModelProvider.OPENAI,
        {"model": "gpt-4o", "messages": [], "presence_penalty": 0.9},
    )
    assert "presence_penalty" not in params


@pytest.mark.asyncio
async def test_short_content_skips_llm_chunking():
    agent = _make_agent()
    result = await agent.generate_chunks("short board post")
    assert result["chunks"] == ["short board post"]
    assert result["summary"] == "short board post"
    assert "thread_id" in result


@pytest.mark.asyncio
async def test_short_batch_skips_llm_chunking():
    agent = _make_agent()
    results = await agent.generate_chunks_batch(["one", "two", "three"])
    assert [item["chunks"] for item in results] == [["one"], ["two"], ["three"]]


@pytest.mark.asyncio
async def test_safe_model_call_retries_without_temperature():
    agent = _make_agent()

    class Completions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'message': \"Unsupported value: 'temperature' does not support 0.3 with this model.\", 'param': 'temperature', 'code': 'unsupported_value'}}"
                )
            return {"ok": True}

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {})()
            self.chat.completions = Completions()

    client = Client()
    result = await agent._safe_model_call(
        client,
        {"model": "gpt-4o", "messages": [], "temperature": 0.3},
    )
    assert result == {"ok": True}
    assert len(client.chat.completions.calls) == 2
    assert "temperature" not in client.chat.completions.calls[1]
    assert "gpt-4o" in agent._models_without_temperature
