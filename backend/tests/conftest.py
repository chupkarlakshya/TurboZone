"""Keep tests off the real database file: every test session gets a throwaway in-memory store."""
import pytest

from backend import app as appmod
from backend.pipeline.store import Store


@pytest.fixture(autouse=True)
def _memory_store():
    appmod.set_store(Store(":memory:"))
    yield
    appmod.set_store(None)
