import json
import logging

from gaugevision.logging_config import JsonFormatter, configure_logging


def test_json_formatter_produces_valid_json_with_core_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = formatter.format(record)
    payload = json.loads(line)  # must not raise

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.verdict = "GO"
    record.inference_ms = 12.5

    payload = json.loads(formatter.format(record))
    assert payload["verdict"] == "GO"
    assert payload["inference_ms"] == 12.5


def test_json_formatter_includes_exception_traceback():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_attaches_single_stream_handler():
    configure_logging()
    configure_logging()  # idempotent-ish: should not duplicate handlers
    root = logging.getLogger()
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) == 1
    assert isinstance(stream_handlers[0].formatter, JsonFormatter)
