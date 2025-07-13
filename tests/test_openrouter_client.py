import os
import sys
import types
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules.setdefault('pandas', types.ModuleType('pandas'))
sys.modules.setdefault('nest_asyncio', types.ModuleType('nest_asyncio'))
sys.modules.setdefault('IPython', types.ModuleType('IPython'))
sys.modules.setdefault('numpy', types.ModuleType('numpy'))
sys.modules.setdefault('yaml', types.ModuleType('yaml'))

from knowledge_agents.model_ops import KnowledgeAgent, ModelProvider

@pytest.mark.asyncio
async def test_openrouter_client_initialization(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    agent = KnowledgeAgent()
    client = await agent._create_client(ModelProvider.OPENROUTER)
    assert client is not None
    # verify base_url is set correctly
    assert getattr(client, "base_url", None) == "https://openrouter.ai/api/v1"

