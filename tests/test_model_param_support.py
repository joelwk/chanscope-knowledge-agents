from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider


def _make_agent():
    # Avoid KnowledgeAgent.__init__ (requires API keys / client init).
    agent = KnowledgeAgent.__new__(KnowledgeAgent)
    agent._unsupported_params_by_model = {}
    agent._models_without_temperature = set()
    return agent


def test_prepare_model_params_filters_penalties_for_gpt5():
    agent = _make_agent()
    params = agent._prepare_model_params(
        ModelProvider.OPENAI,
        {"model": "gpt-5.2-2025-12-11", "messages": []},
    )
    assert "temperature" in params
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


def test_cached_unsupported_params_are_removed():
    agent = _make_agent()
    agent._mark_unsupported_params_for_model("gpt-4o", ["presence_penalty"])

    params = agent._prepare_model_params(
        ModelProvider.OPENAI,
        {"model": "gpt-4o", "messages": [], "presence_penalty": 0.9},
    )
    assert "presence_penalty" not in params
