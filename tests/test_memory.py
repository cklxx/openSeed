"""Tests for research memory persistence (MemoryStore)."""

from __future__ import annotations

import pytest

from openseed.agent.memory import MemoryStore
from openseed.storage.library import PaperLibrary


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    lib = PaperLibrary(tmp_path / "lib")
    return MemoryStore(lib)


class TestSaveAndRetrieve:
    def test_save_returns_id(self, store: MemoryStore) -> None:
        rid = store.save_memory("s1", "user", "hello world")
        assert isinstance(rid, int) and rid > 0

    def test_get_session_history(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "first message")
        store.save_memory("s1", "assistant", "reply")
        store.save_memory("s2", "user", "other session")
        history = store.get_session_history("s1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_session_history_ordered_by_time(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "first")
        store.save_memory("s1", "user", "second")
        history = store.get_session_history("s1")
        assert history[0].content == "first"
        assert history[1].content == "second"


class TestFTSSearch:
    def test_search_finds_matching(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "transformer architecture attention")
        store.save_memory("s1", "user", "cooking recipes for dinner")
        results = store.search_memories("transformer attention")
        assert len(results) >= 1
        assert any("transformer" in r.content for r in results)

    def test_search_no_match(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "hello world")
        results = store.search_memories("zzzznonexistent")
        assert results == []

    def test_search_respects_top_k(self, store: MemoryStore) -> None:
        for i in range(5):
            store.save_memory("s1", "user", f"neural network model {i}")
        results = store.search_memories("neural network", top_k=3)
        assert len(results) <= 3

    def test_search_empty_query_returns_empty(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "some content")
        assert store.search_memories("") == []
        assert store.search_memories("   ") == []

    def test_search_special_chars_fallback(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "test content with special chars")
        results = store.search_memories("test (content")
        assert isinstance(results, list)


class TestTopics:
    def test_topics_roundtrip(self, store: MemoryStore) -> None:
        topics = ["ml", "transformers", "attention"]
        store.save_memory("s1", "user", "about attention", topics=topics)
        history = store.get_session_history("s1")
        assert history[0].topics == topics

    def test_topics_none_stored_as_null(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "no topics")
        history = store.get_session_history("s1")
        assert history[0].topics is None


class TestClearSession:
    def test_clear_returns_count(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "msg1")
        store.save_memory("s1", "user", "msg2")
        assert store.clear_session("s1") == 2

    def test_clear_removes_from_history(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "msg")
        store.clear_session("s1")
        assert store.get_session_history("s1") == []

    def test_clear_nonexistent_returns_zero(self, store: MemoryStore) -> None:
        assert store.clear_session("nope") == 0

    def test_clear_does_not_affect_other_sessions(self, store: MemoryStore) -> None:
        store.save_memory("s1", "user", "session 1")
        store.save_memory("s2", "user", "session 2")
        store.clear_session("s1")
        assert len(store.get_session_history("s2")) == 1
