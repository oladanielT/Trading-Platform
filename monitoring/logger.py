"""
Thread-safe monitoring logger for the trading platform.

Responsibilities:
- Log trades, errors, and system events in a structured (key/value) style.
- Configurable log levels and handlers (console, rotating file).
- Thread-safe configuration and simple in-memory recent-log cache for fast access.
- Avoids logging sensitive fields if present in structured payloads.

Usage:
    from monitoring.logger import LoggerManager, scrub_secrets

    mgr = LoggerManager()
    mgr.configure(level="INFO", log_file="logs/trading.log")

    mgr.log_event("system.start", {"msg": "service started"})
    mgr.log_trade({"id": "t1", "symbol": "BTC/USDT", "price": 60000, "size": 0.001})
    try:
        ...
    except Exception:
        mgr.log_error("order_execution_failed", exc_info=True)
"""
from __future__ import annotations

import logging
import logging.handlers
import threading
import json
import time
from collections import deque
from typing import Any, Dict, Optional, Iterable, List

# Fields considered sensitive and should be scrubbed from structured payloads
_DEFAULT_SENSITIVE_KEYS = frozenset({"api_key", "secret", "password", "private_key", "token"})

# Default in-memory cache size for recent structured logs
_DEFAULT_RECENT_CACHE_SIZE = 200


def _as_kv_str(event: str, payload: Optional[Dict[str, Any]]) -> str:
    """Return a compact key/value string for console/file logs."""
    if not payload:
        return event
    try:
        # Use JSON for structured payloads (compact)
        return f"{event} {json.dumps(payload, separators=(',', ':'), default=str)}"
    except Exception:
        return f"{event} {str(payload)}"


def scrub_secrets(payload: Optional[Dict[str, Any]], sensitive_keys: Optional[Iterable[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Return a shallow copy of payload with sensitive fields redacted.

    Args:
        payload: structured payload dictionary (may be None)
        sensitive_keys: optional iterable of keys to redact (case-insensitive)
    """
    if payload is None:
        return None
    keys = set(k.lower() for k in (sensitive_keys or _DEFAULT_SENSITIVE_KEYS))
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if k.lower() in keys:
            out[k] = "<REDACTED>"
        else:
            out[k] = v
    return out


class LoggerManager:
    """
    Thread-safe logger manager.

    Provides:
    - configure(level, log_file, max_bytes, backup_count)
    - log_trade(trade_payload)
    - log_error(event, message, exc_info)
    - log_event(event, payload)

    Maintains a recent in-memory deque of structured logs for quick inspection.
    """

    def __init__(self, name: str = "trading.monitor"):
        self._name = name
        self._lock = threading.RLock()
        self._logger = logging.getLogger(self._name)
        # Default to WARNING until configured
        self._logger.setLevel(logging.WARNING)
        # Prevent double handlers in interactive re-imports
        self._logger.propagate = False
        self._handlers_attached = False

        # Recent structured logs (thread-safe via _lock)
        self._recent = deque(maxlen=_DEFAULT_RECENT_CACHE_SIZE)

    # -----------------------
    # Configuration / Setup
    # -----------------------
    def configure(
        self,
        level: str | int = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console: bool = True,
    ) -> None:
        """
        Configure logger handlers and level (thread-safe).

        Args:
            level: logging level name or int
            log_file: optional path to rotating log file
            max_bytes: rotation size in bytes
            backup_count: number of rotated files to keep
            console: enable console handler
        """
        with self._lock:
            # Normalize level
            lvl = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
            self._logger.setLevel(lvl)

            # Remove existing handlers if re-configuring
            for h in list(self._logger.handlers):
                try:
                    self._logger.removeHandler(h)
                    h.close()
                except Exception:
                    pass

            handlers: List[logging.Handler] = []
            if console:
                ch = logging.StreamHandler()
                ch.setLevel(lvl)
                ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                handlers.append(ch)

            if log_file:
                fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
                fh.setLevel(lvl)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                handlers.append(fh)

            for h in handlers:
                self._logger.addHandler(h)

            self._handlers_attached = True
            # Log the configuration event without secrets
            self._logger.debug("logger.configured", extra={"level": lvl, "log_file": bool(log_file)})

    def add_handler(self, handler: logging.Handler) -> None:
        """Attach a custom handler (thread-safe)."""
        with self._lock:
            self._logger.addHandler(handler)
            self._handlers_attached = True

    # -----------------------
    # Internal helpers
    # -----------------------
    def _record_recent(self, kind: str, event: str, payload: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self._recent.appendleft({"ts": int(time.time()), "kind": kind, "event": event, "payload": payload})

    def get_recent(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a snapshot list of recent structured logs (thread-safe)."""
        with self._lock:
            if limit is None:
                return list(self._recent)
            return list(self._recent)[:limit]

    # -----------------------
    # Public logging methods
    # -----------------------
    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None, level: str | int = "INFO") -> None:
        """
        Log a generic system event with structured payload.

        Args:
            event: short event name, e.g. "system.start"
            payload: optional structured payload (will be scrubbed)
            level: log level
        """
        payload_safe = scrub_secrets(payload)
        msg = _as_kv_str(event, payload_safe)
        lvl = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
        # Keep structured data in extra for downstream handlers
        self._logger.log(lvl, msg, extra={"event": event, "payload": payload_safe})
        self._record_recent("event", event, payload_safe)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log a trade (structured).

        Trade dict should include at least: id, symbol, price, size, side (optional).
        """
        trade_safe = scrub_secrets(trade)
        msg = _as_kv_str("trade", trade_safe)
        # Use INFO for trades by default
        self._logger.info(msg, extra={"event": "trade", "payload": trade_safe})
        self._record_recent("trade", "trade", trade_safe)

    def log_error(self, event: str, message: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, exc_info: Any = None) -> None:
        """
        Log an error or exception.

        Args:
            event: short error event name
            message: human-readable message
            payload: optional structured payload
            exc_info: exception info (True, exception tuple, or False)
        """
        payload_safe = scrub_secrets(payload)
        full_msg = f"{event} {message or ''}".strip()
        msg = _as_kv_str(full_msg, payload_safe)
        # Use ERROR level by default
        self._logger.error(msg, exc_info=exc_info, extra={"event": event, "payload": payload_safe})
        self._record_recent("error", event, {"message": message, "payload": payload_safe})

    # -----------------------
    # Utilities
    # -----------------------
    def get_logger(self) -> logging.Logger:
        """Return underlying logger instance (for advanced use)."""
        return self._logger


# Module-level default manager for convenience
_default_manager = LoggerManager()


def configure(level: str | int = "INFO", log_file: Optional[str] = None, **kwargs) -> None:
    """Convenience configure on module-level manager."""
    _default_manager.configure(level=level, log_file=log_file, **kwargs)


def log_event(event: str, payload: Optional[Dict[str, Any]] = None, level: str | int = "INFO") -> None:
    _default_manager.log_event(event, payload, level)


def log_trade(trade: Dict[str, Any]) -> None:
    _default_manager.log_trade(trade)


def log_error(event: str, message: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, exc_info: Any = None) -> None:
    _default_manager.log_error(event, message, payload, exc_info)


# Example usage
if __name__ == "__main__":
    import time as _t

    configure(level="DEBUG", log_file=None)
    log_event("system.start", {"msg": "monitor starting", "version": "0.1.0"})
    log_trade({"id": "tx-123", "symbol": "ETH/USDT", "price": 1800.5, "size": 0.25, "side": "buy"})
    try:
        1 / 0
    except Exception:
        log_error("math.failure", "division by zero in demo", exc_info=True)
    _t.sleep(0.1)
    print("Recent logs:", _default_manager.get_recent(5))


def _as_kv_str(event: str, payload: Optional[Dict[str, Any]]) -> str:
    """Return a compact key/value string for console/file logs."""
    if not payload:
        return event
    try:
        # Use JSON for structured payloads (compact)
        return f"{event} {json.dumps(payload, separators=(',', ':'), default=str)}"
    except Exception:
        return f"{event} {str(payload)}"


def scrub_secrets(payload: Optional[Dict[str, Any]], sensitive_keys: Optional[Iterable[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Return a shallow copy of payload with sensitive fields redacted.

    Args:
        payload: structured payload dictionary (may be None)
        sensitive_keys: optional iterable of keys to redact (case-insensitive)
    """
    if payload is None:
        return None
    keys = set(k.lower() for k in (sensitive_keys or _DEFAULT_SENSITIVE_KEYS))
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        if k.lower() in keys:
            out[k] = "<REDACTED>"
        else:
            out[k] = v
    return out


class LoggerManager:
    """
    Thread-safe logger manager.

    Provides:
    - configure(level, log_file, max_bytes, backup_count)
    - log_trade(trade_payload)
    - log_error(event, message, exc_info)
    - log_event(event, payload)

    Maintains a recent in-memory deque of structured logs for quick inspection.
    """

    def __init__(self, name: str = "trading.monitor"):
        self._name = name
        self._lock = threading.RLock()
        self._logger = logging.getLogger(self._name)
        # Default to WARNING until configured
        self._logger.setLevel(logging.WARNING)
        # Prevent double handlers in interactive re-imports
        self._logger.propagate = False
        self._handlers_attached = False

        # Recent structured logs (thread-safe via _lock)
        self._recent = deque(maxlen=_DEFAULT_RECENT_CACHE_SIZE)

    # -----------------------
    # Configuration / Setup
    # -----------------------
    def configure(
        self,
        level: str | int = "INFO",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console: bool = True,
    ) -> None:
        """
        Configure logger handlers and level (thread-safe).

        Args:
            level: logging level name or int
            log_file: optional path to rotating log file
            max_bytes: rotation size in bytes
            backup_count: number of rotated files to keep
            console: enable console handler
        """
        with self._lock:
            # Normalize level
            lvl = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
            self._logger.setLevel(lvl)

            # Remove existing handlers if re-configuring
            for h in list(self._logger.handlers):
                try:
                    self._logger.removeHandler(h)
                    h.close()
                except Exception:
                    pass

            handlers: List[logging.Handler] = []
            if console:
                ch = logging.StreamHandler()
                ch.setLevel(lvl)
                ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                handlers.append(ch)

            if log_file:
                fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
                fh.setLevel(lvl)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                handlers.append(fh)

            for h in handlers:
                self._logger.addHandler(h)

            self._handlers_attached = True
            # Log the configuration event without secrets
            self._logger.debug("logger.configured", extra={"level": lvl, "log_file": bool(log_file)})

    def add_handler(self, handler: logging.Handler) -> None:
        """Attach a custom handler (thread-safe)."""
        with self._lock:
            self._logger.addHandler(handler)
            self._handlers_attached = True

    # -----------------------
    # Internal helpers
    # -----------------------
    def _record_recent(self, kind: str, event: str, payload: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self._recent.appendleft({"ts": int(time.time()), "kind": kind, "event": event, "payload": payload})

    def get_recent(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a snapshot list of recent structured logs (thread-safe)."""
        with self._lock:
            if limit is None:
                return list(self._recent)
            return list(self._recent)[:limit]

    # -----------------------
    # Public logging methods
    # -----------------------
    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None, level: str | int = "INFO") -> None:
        """
        Log a generic system event with structured payload.

        Args:
            event: short event name, e.g. "system.start"
            payload: optional structured payload (will be scrubbed)
            level: log level
        """
        payload_safe = scrub_secrets(payload)
        msg = _as_kv_str(event, payload_safe)
        lvl = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)
        # Keep structured data in extra for downstream handlers
        self._logger.log(lvl, msg, extra={"event": event, "payload": payload_safe})
        self._record_recent("event", event, payload_safe)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log a trade (structured).

        Trade dict should include at least: id, symbol, price, size, side (optional).
        """
        trade_safe = scrub_secrets(trade)
        msg = _as_kv_str("trade", trade_safe)
        # Use INFO for trades by default
        self._logger.info(msg, extra={"event": "trade", "payload": trade_safe})
        self._record_recent("trade", "trade", trade_safe)

    def log_error(self, event: str, message: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, exc_info: Any = None) -> None:
        """
        Log an error or exception.

        Args:
            event: short error event name
            message: human-readable message
            payload: optional structured payload
            exc_info: exception info (True, exception tuple, or False)
        """
        payload_safe = scrub_secrets(payload)
        full_msg = f"{event} {message or ''}".strip()
        msg = _as_kv_str(full_msg, payload_safe)
        # Use ERROR level by default
        self._logger.error(msg, exc_info=exc_info, extra={"event": event, "payload": payload_safe})
        self._record_recent("error", event, {"message": message, "payload": payload_safe})

    # -----------------------
    # Utilities
    # -----------------------
    def get_logger(self) -> logging.Logger:
        """Return underlying logger instance (for advanced use)."""
        return self._logger


# Module-level default manager for convenience
_default_manager = LoggerManager()


def configure(level: str | int = "INFO", log_file: Optional[str] = None, **kwargs) -> None:
    """Convenience configure on module-level manager."""
    _default_manager.configure(level=level, log_file=log_file, **kwargs)


def log_event(event: str, payload: Optional[Dict[str, Any]] = None, level: str | int = "INFO") -> None:
    _default_manager.log_event(event, payload, level)


def log_trade(trade: Dict[str, Any]) -> None:
    _default_manager.log_trade(trade)


def log_error(event: str, message: Optional[str] = None, payload: Optional[Dict[str, Any]] = None, exc_info: Any = None) -> None:
    _default_manager.log_error(event, message, payload, exc_info)


# Example usage
if __name__ == "__main__":
    import time as _t

    configure(level="DEBUG", log_file=None)
    log_event("system.start", {"msg": "monitor starting", "version": "0.1.0"})
    log_trade({"id": "tx-123", "symbol": "ETH/USDT", "price": 1800.5, "size": 0.25, "side": "buy"})
    try:
        1 / 0
    except Exception:
        log_error("math.failure", "division by zero in demo", exc_info=True)
    _t.sleep(0.1)
    print("Recent logs:", _default_manager.get_recent(5))