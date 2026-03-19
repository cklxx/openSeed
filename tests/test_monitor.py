"""Tests for openseed.monitor — openMax integration with graceful degradation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from openseed.monitor import get_usage_summary, make_usage_recorder, record_research_lesson


class TestMakeUsageRecorder:
    def test_returns_callable(self) -> None:
        recorder = make_usage_recorder("summarize")
        assert callable(recorder)

    def test_callable_returns_none(self) -> None:
        recorder = make_usage_recorder("summarize")
        result = recorder(MagicMock())
        assert result is None

    def test_degrades_gracefully_when_openmax_missing(self) -> None:
        with patch.dict(sys.modules, {"openmax": None, "openmax.usage": None}):
            recorder = make_usage_recorder("test_op")
            recorder(MagicMock())

    def test_calls_usage_store_when_openmax_available(self) -> None:
        mock_usage_store = MagicMock()
        mock_usage_from_result = MagicMock(return_value="usage_record")
        mock_openmax_usage = MagicMock(
            UsageStore=MagicMock(return_value=mock_usage_store),
            usage_from_result=mock_usage_from_result,
        )
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            recorder = make_usage_recorder("summarize")
            fake_result = MagicMock()
            recorder(fake_result)
        mock_usage_store.save.assert_called_once()

    def test_op_name_appears_in_session_id(self) -> None:
        saved_session_ids: list[str] = []

        def capture_save(record):
            saved_session_ids.append(record)

        mock_store = MagicMock()
        mock_store.save.side_effect = capture_save
        mock_usage_from_result = MagicMock(side_effect=lambda sid, _: sid)
        mock_openmax_usage = MagicMock(
            UsageStore=MagicMock(return_value=mock_store),
            usage_from_result=mock_usage_from_result,
        )
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            recorder = make_usage_recorder("myop")
            recorder(MagicMock())
        assert any("myop" in str(s) for s in saved_session_ids)


class TestRecordResearchLesson:
    def test_does_not_raise_when_openmax_missing(self) -> None:
        mods = {"openmax": None, "openmax.memory": None, "openmax.memory.store": None}
        with patch.dict(sys.modules, mods):
            record_research_lesson("attention", "lesson text")

    def test_calls_memory_store_when_available(self) -> None:
        mock_store = MagicMock()
        mock_memory_store_module = MagicMock(MemoryStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.memory.store": mock_memory_store_module}):
            record_research_lesson("attention mechanisms", "key insight", cwd="/tmp")
        mock_store.record_lesson.assert_called_once()

    def test_passes_topic_in_task_field(self) -> None:
        mock_store = MagicMock()
        mock_memory_store_module = MagicMock(MemoryStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.memory.store": mock_memory_store_module}):
            record_research_lesson("transformers", "insight", cwd="/tmp")
        call_kwargs = mock_store.record_lesson.call_args.kwargs
        assert "transformers" in call_kwargs.get("task", "")

    def test_uses_home_dir_when_cwd_not_given(self) -> None:
        mock_store = MagicMock()
        mock_memory_store_module = MagicMock(MemoryStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.memory.store": mock_memory_store_module}):
            record_research_lesson("topic", "insight")
        mock_store.record_lesson.assert_called_once()

    def test_ignores_exception_from_store(self) -> None:
        mock_store = MagicMock()
        mock_store.record_lesson.side_effect = RuntimeError("store failed")
        mock_memory_store_module = MagicMock(MemoryStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.memory.store": mock_memory_store_module}):
            record_research_lesson("topic", "insight")


class TestGetUsageSummary:
    def test_returns_none_when_openmax_missing(self) -> None:
        with patch.dict(sys.modules, {"openmax": None, "openmax.usage": None}):
            result = get_usage_summary()
        assert result is None

    def test_returns_none_when_no_openseed_records(self) -> None:
        mock_store = MagicMock()
        mock_store.list_all.return_value = []
        mock_openmax_usage = MagicMock(UsageStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            result = get_usage_summary()
        assert result is None

    def test_returns_summary_string_when_records_exist(self) -> None:
        fake_record = MagicMock()
        fake_record.session_id = "openseed_summarize_abc12345"
        mock_agg = MagicMock()
        mock_agg.summary_line.return_value = "$0.01 / 1000 tokens"
        mock_store = MagicMock()
        mock_store.list_all.return_value = [fake_record]
        mock_store.aggregate.return_value = mock_agg
        mock_openmax_usage = MagicMock(UsageStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            result = get_usage_summary()
        assert result is not None
        assert "1 ops" in result

    def test_ignores_non_openseed_records(self) -> None:
        other_record = MagicMock()
        other_record.session_id = "someother_tool_12345"
        mock_store = MagicMock()
        mock_store.list_all.return_value = [other_record]
        mock_openmax_usage = MagicMock(UsageStore=MagicMock(return_value=mock_store))
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            result = get_usage_summary()
        assert result is None

    def test_returns_none_on_unexpected_exception(self) -> None:
        mock_openmax_usage = MagicMock()
        mock_openmax_usage.UsageStore.side_effect = RuntimeError("broken")
        with patch.dict(sys.modules, {"openmax.usage": mock_openmax_usage}):
            result = get_usage_summary()
        assert result is None
