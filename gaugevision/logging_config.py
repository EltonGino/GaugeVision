"""Structured logging — CLAUDE.md §7 Phase 5.

Plain Python ``logging`` with a JSON line formatter — deliberately not a
heavyweight observability stack (no OpenTelemetry, no external log
shipper). Each log line is one JSON object, so it's directly parseable by
any downstream tool (a log aggregator, `jq`, a simple grep) without a
custom parser.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Include any structured extras passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and key not in payload:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single JSON-formatted stream handler to the root logger.

    Idempotent-ish: clears any handlers this call previously added so
    re-invoking (e.g. across module reloads in a notebook) doesn't
    duplicate log lines.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
