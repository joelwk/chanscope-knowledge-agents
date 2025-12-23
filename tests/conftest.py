import asyncio
import inspect

import pytest


def _has_pytest_asyncio(config: pytest.Config) -> bool:
    """Check if pytest-asyncio is available and loaded."""
    return config.pluginmanager.hasplugin("pytest_asyncio")


def pytest_configure(config: pytest.Config) -> None:
    # Ensure the asyncio marker is always registered, even without pytest-asyncio.
    config.addinivalue_line("markers", "asyncio: mark test as asyncio")
    config._codex_asyncio_fallback = not _has_pytest_asyncio(config)


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    """Fallback executor for async tests when pytest-asyncio isn't available."""
    if not getattr(pyfuncitem.config, "_codex_asyncio_fallback", False):
        return False

    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return False

    funcargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(pyfuncitem.obj(**funcargs))
    return True
