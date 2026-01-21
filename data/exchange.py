"""
Hardened async CCXT exchange wrapper (minimal, focused on data integrity).

Guarantees provided:
- Connection lifecycle is guarded (prevent concurrent connect())
- disconnect() is idempotent
- Per-request timeout via asyncio.wait_for
- Retry only on explicit network-like errors (no broad Exception)
- Adaptive backoff for rate-limit events (RateLimitExceeded)
- fetch_ohlcv_range returns deduplicated, ascending OHLCV candles (timestamps in ms)
- All returned data is normalized: timestamps (ms ints), prices/volume floats, symbol normalized
- Validates OHLCV schema on receipt and fails loudly on bad data
- health_check() verifies connectivity and market availability
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from threading import Lock

# Keep ccxt import guarded so module can be imported in test contexts
try:
    import ccxt.async_support as ccxt_async  # type: ignore
    import ccxt  # type: ignore
except Exception:
    ccxt_async = None
    ccxt = None

try:
    import numpy as np
except ImportError:
    np = None


# Configure logger
logger = logging.getLogger(__name__)


class ExchangeError(RuntimeError):
    pass


class AuthenticationError(ExchangeError):
    pass


class RateLimitError(ExchangeError):
    pass


class DataValidationError(ExchangeError):
    pass


class ExchangeConnector:
    """
    Minimal hardened exchange wrapper.

    Usage:
        conn = ExchangeConnector(exchange_id='binance', api_key=..., secret=..., mode='testnet')
        await conn.connect()
        candles = await conn.fetch_ohlcv_range(symbol, timeframe, since_ms, until_ms)
        await conn.disconnect()

    Guarantees:
    - fetch_ohlcv_range returns list[dict] sorted ascending by timestamp (ms), deduplicated by timestamp.
    - All dicts: {"symbol": str, "timestamp": int(ms), "open": float, "high": float, "low": float, "close": float, "volume": float}
    - Raises explicit exceptions on invalid data, timeouts, or unrecoverable exchange errors.
    """

    DEFAULT_TIMEOUT = 10.0
    DEFAULT_PAGE_LIMIT = 500

    def __init__(self, exchange_id: str, api_key: Optional[str] = None, secret: Optional[str] = None, mode: str = "live"):
        if ccxt_async is None:
            raise ImportError("ccxt.async_support required by ExchangeConnector")
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.mode = mode.lower()
        self._lock = Lock()
        self._client = None
        self._connected = False
        self._last_health = 0.0
        # rate limit backoff state
        self._rate_backoff = 1.0

    # -----------------------
    # Lifecycle
    # -----------------------
    async def connect(self) -> None:
        """Establish connection to exchange. Prevent concurrent connects."""
        if not self._lock.acquire(blocking=False):
            raise ExchangeError("connect() already in progress")
        try:
            if self._connected:
                return
            ex_cls = getattr(ccxt_async, self.exchange_id, None) or getattr(ccxt_async, self.exchange_id.capitalize(), None)
            if ex_cls is None:
                # fallback to ccxt provided mapping
                try:
                    ex_cls = getattr(ccxt_async, self.exchange_id)
                except Exception:
                    raise ExchangeError(f"Exchange '{self.exchange_id}' not available in ccxt.async_support")
            # instantiate client
            client = ex_cls({
                "apiKey": self.api_key,
                "secret": self.secret,
                "enableRateLimit": True,
            })
            # try to set common testnet flags if requested
            if self.mode == "testnet":
                # explicit sandbox for testnet when user requests it
                try:
                    if hasattr(client, "set_sandbox_mode"):
                        client.set_sandbox_mode(True)
                    if "test" in client.urls and client.urls.get("test"):
                        client.urls["api"] = client.urls["test"]
                except Exception:
                    pass
            elif self.mode == "paper":
                # paper trading should use live market data; keep sandbox off
                try:
                    if hasattr(client, "set_sandbox_mode"):
                        client.set_sandbox_mode(False)
                except Exception:
                    pass

            # verify basic capability: fetchOHLCV
            if not getattr(client, "has", {}).get("fetchOHLCV", False):
                # fail early: we require OHLCV for this connector
                await client.close()
                raise ExchangeError(f"Exchange '{self.exchange_id}' does not support fetchOHLCV")

            self._client = client
            self._connected = True
            self._last_health = time.time()
        finally:
            self._lock.release()

    async def disconnect(self) -> None:
        """Disconnect (idempotent)."""
        if not self._connected:
            return
        # safe close
        try:
            if self._client:
                try:
                    await self._client.close()
                except Exception:
                    # swallow close errors but mark disconnected
                    pass
        finally:
            self._client = None
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -----------------------
    # Helpers: errors selection
    # Only retry network-like errors from ccxt; do not retry authentication/logical errors
    # -----------------------
    def _is_retryable_exception(self, exc: Exception) -> bool:
        """Return True if exception is network-like and safe to retry."""
        # Prefer known ccxt network exceptions if available
        if ccxt is not None:
            retryable = (
                getattr(ccxt, "NetworkError", None),
                getattr(ccxt, "RequestTimeout", None),
                getattr(ccxt, "ExchangeNotAvailable", None),
                getattr(ccxt, "DDoSProtection", None),
                getattr(ccxt, "RateLimitExceeded", None),
            )
            for cls in retryable:
                if cls is not None and isinstance(exc, cls):
                    return True
        # fallback: asyncio.TimeoutError considered retryable at wrapper level
        return isinstance(exc, asyncio.TimeoutError)

    def _is_rate_limit_exception(self, exc: Exception) -> bool:
        if ccxt is not None and getattr(ccxt, "RateLimitExceeded", None) is not None:
            return isinstance(exc, getattr(ccxt, "RateLimitExceeded"))
        # best-effort string match fallback
        return "rate limit" in str(exc).lower()

    # -----------------------
    # Request wrapper: timeout + retry + backoff
    # -----------------------
    async def _request_with_retry(self, coro_callable, *args, timeout: Optional[float] = None, max_attempts: int = 5, **kwargs):
        """
        Execute coroutine factory with timeout + retries on network errors only.

        - timeout: per-attempt seconds (default DEFAULT_TIMEOUT)
        - adaptive backoff on rate limit: exponential backoff (base self._rate_backoff)
        - exponential backoff on timeout errors
        - never retries on authentication/logical errors
        """
        timeout = timeout or self.DEFAULT_TIMEOUT
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < max_attempts:
            attempt += 1
            try:
                # apply per-call timeout using asyncio.wait_for
                coro = coro_callable(*args, **kwargs)
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError as exc:
                # Handle timeout specifically with exponential backoff
                last_exc = exc
                if attempt < max_attempts:
                    # Exponential backoff with jitter for timeouts
                    backoff = min(30.0, 1.0 * (2 ** (attempt - 1))) + (0.1 * (attempt % 5))
                    logger.warning(
                        f"Request timeout (attempt {attempt}/{max_attempts}), "
                        f"retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Request timeout after {max_attempts} attempts")
                    raise
            except Exception as exc:
                last_exc = exc
                # if rate-limit, backoff and retry
                if self._is_rate_limit_exception(exc):
                    # increase backoff
                    self._rate_backoff = min(60.0, (self._rate_backoff or 1.0) * 2.0)
                    logger.warning(
                        f"Rate limit exceeded (attempt {attempt}/{max_attempts}), "
                        f"backing off {self._rate_backoff:.2f}s..."
                    )
                    await asyncio.sleep(self._rate_backoff)
                    continue
                # if retryable network error, small jittered backoff
                if self._is_retryable_exception(exc):
                    sleep_for = min(10.0, 0.5 * (2 ** (attempt - 1))) + (0.01 * (attempt % 3))
                    logger.warning(
                        f"Network error (attempt {attempt}/{max_attempts}): {exc}, "
                        f"retrying in {sleep_for:.2f}s..."
                    )
                    await asyncio.sleep(sleep_for)
                    continue
                # do not retry authentication or logical errors
                raise
        # exhausted attempts
        raise last_exc or ExchangeError("request failed after retries")

    # -----------------------
    # OHLCV Normalization & Validation
    # -----------------------
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol to 'BASE/QUOTE' uppercase format (best-effort)."""
        if not isinstance(symbol, str):
            raise DataValidationError("symbol must be a string")
        s = symbol.replace("-", "/").replace("_", "/").upper()
        if "/" not in s:
            # try split uppercase letters fallback
            return s
        base, quote = s.split("/", 1)
        return f"{base.strip()}/{quote.strip()}"

    @staticmethod
    def _to_ms(ts: Any) -> int:
        """Coerce timestamp to milliseconds integer."""
        if isinstance(ts, int):
            # assume ms if > 1e12 (unlikely) else if seconds convert
            if ts > 10 ** 12:
                return int(ts)
            if ts > 10 ** 9:
                # seconds -> ms
                return int(ts * 1000)
            return int(ts)
        if isinstance(ts, float):
            if ts > 1e12:
                return int(ts)
            if ts > 1e9:
                return int(ts * 1000)
            return int(ts)
        raise DataValidationError(f"Invalid timestamp type: {type(ts)}")

    @staticmethod
    def _validate_and_normalize_ohlcv(raw: Sequence[Any], symbol: str) -> Dict[str, Any]:
        """
        Accepts either sequence-like OHLCV ([ts, o, h, l, c, v]) or dict and returns normalized dict.

        Raises DataValidationError on invalid shape/types.
        """
        if isinstance(raw, dict):
            ts = raw.get("timestamp") or raw.get("time") or raw.get("date") or raw.get("ts")
            o = raw.get("open")
            h = raw.get("high")
            l = raw.get("low")
            c = raw.get("close")
            v = raw.get("volume") or raw.get("amount")
        elif isinstance(raw, (list, tuple)):
            if len(raw) < 5:
                raise DataValidationError("OHLCV entry must have at least 5 elements")
            ts = raw[0]
            o = raw[1]
            h = raw[2]
            l = raw[3]
            c = raw[4]
            v = raw[5] if len(raw) > 5 else 0.0
        else:
            raise DataValidationError("Unsupported OHLCV entry type")

        ts_ms = ExchangeConnector._to_ms(ts)
        try:
            o_f = float(o)
            h_f = float(h)
            l_f = float(l)
            c_f = float(c)
            v_f = float(v)
        except Exception:
            raise DataValidationError("OHLCV prices/volume must be numeric")

        if not (0 <= l_f <= h_f):
            # allow equality but not nonsensical values
            raise DataValidationError("Invalid OHLCV values: low > high or negative")

        return {
            "symbol": ExchangeConnector._normalize_symbol(symbol),
            "timestamp": int(ts_ms),
            "open": o_f,
            "high": h_f,
            "low": l_f,
            "close": c_f,
            "volume": v_f,
        }

    # -----------------------
    # Synthetic data generation (for paper mode)
    # -----------------------
    def _generate_synthetic_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic OHLCV data for paper trading.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d', etc.)
            since_ms: Start timestamp in milliseconds
            until_ms: End timestamp in milliseconds
            limit: Number of candles to generate
            
        Returns:
            List of OHLCV dicts sorted by timestamp
        """
        if np is None:
            raise ImportError("numpy required for synthetic data generation")
        
        # Base prices for different symbols
        base_prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2300.0,
            'BNB/USDT': 310.0,
            'XRP/USDT': 0.62,
            'ADA/USDT': 0.48,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Timeframe to seconds mapping
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        candle_seconds = timeframe_seconds.get(timeframe, 3600)
        
        # Determine time range
        now_ms = int(time.time() * 1000)
        if until_ms is None:
            until_ms = now_ms
        if since_ms is None:
            since_ms = until_ms - (limit * candle_seconds * 1000)
        
        # Generate timestamps
        num_candles = max(1, (until_ms - since_ms) // (candle_seconds * 1000))
        num_candles = min(num_candles, limit)
        
        timestamps_ms = [
            since_ms + (i * candle_seconds * 1000)
            for i in range(num_candles)
        ]
        
        # Generate price movement (random walk with slight upward trend)
        np.random.seed(hash(symbol) % (2**32))  # Consistent seed per symbol
        returns = np.random.normal(0.0002, 0.02, num_candles)  # Slight upward trend
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        results = []
        for ts_ms, close_price in zip(timestamps_ms, prices):
            volatility = 0.01
            open_price = close_price * (1 + np.random.normal(0, volatility))
            high = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility)))
            low = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility)))
            volume = float(np.random.lognormal(10, 2))
            
            results.append({
                "symbol": self._normalize_symbol(symbol),
                "timestamp": int(ts_ms),
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close_price),
                "volume": volume,
            })
        
        logger.info(
            f"Generated {len(results)} synthetic {timeframe} candles for {symbol} "
            f"(${results[0]['close']:.2f} → ${results[-1]['close']:.2f})"
        )
        
        return results

    # -----------------------
    # fetch_ohlcv_range
    # -----------------------
    async def fetch_ohlcv_range(
        self,
        symbol: str,
        timeframe: str,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        page_limit: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve OHLCV candles for [since_ms .. until_ms) (ms timestamps).

        Guarantees:
        - Ascending timestamp order
        - No duplicate timestamps
        - Validated and normalized records (see _validate_and_normalize_ohlcv)
        - Stops cleanly if exchange returns partial data

        Failure modes:
        - Raises DataValidationError on malformed data
        - Raises ExchangeError or underlying ccxt errors on connectivity/auth issues
        
        Note: Always uses real market data from exchange API, regardless of mode.
        Paper/testnet execution is handled by the Broker, not the data layer.
        """
        if not self._connected or not self._client:
            raise ExchangeError("Not connected")

        limit = int(page_limit or self.DEFAULT_PAGE_LIMIT)
        since = since_ms
        results: List[Dict[str, Any]] = []
        seen_ts = set()
        last_ts = since_ms or -1
        attempts_without_progress = 0
        max_no_progress = 3

        while True:
            # call fetch_ohlcv page
            async def _page_call():
                # ccxt fetch_ohlcv signature: symbol, timeframe, since, limit
                return await self._client.fetch_ohlcv(symbol, timeframe, since, limit)

            page = await self._request_with_retry(_page_call, timeout=timeout)
            if not page:
                # exchange returned empty page -> stop
                break

            # normalize and validate each entry
            page_norm: List[Dict[str, Any]] = []
            for raw in page:
                norm = self._validate_and_normalize_ohlcv(raw, symbol)
                page_norm.append(norm)

            # ensure ascending timestamps in page
            page_ts = [p["timestamp"] for p in page_norm]
            if any(page_ts[i] >= page_ts[i + 1] for i in range(len(page_ts) - 1)):
                # page not strictly ascending -> sort by timestamp
                page_norm.sort(key=lambda x: x["timestamp"])
                page_ts = [p["timestamp"] for p in page_norm]

            # deduplicate by timestamp and append only new ones
            added = 0
            for p in page_norm:
                ts = p["timestamp"]
                if ts in seen_ts:
                    continue
                if until_ms is not None and ts >= until_ms:
                    # we've reached the end window
                    continue
                if since_ms is not None and ts < since_ms:
                    # before requested window -> skip
                    continue
                results.append(p)
                seen_ts.add(ts)
                added += 1
                last_ts = max(last_ts, ts)

            if added == 0:
                attempts_without_progress += 1
            else:
                attempts_without_progress = 0

            # stop if page smaller than limit (likely end) or no progress
            if len(page_norm) < limit:
                break
            if attempts_without_progress >= max_no_progress:
                # prevent infinite loop when exchange keeps returning overlapping pages
                break

            # advance since to last_ts + 1ms so we avoid duplicates
            since = last_ts + 1

        # final sort to guarantee ascending order
        results.sort(key=lambda x: x["timestamp"])
        return results

    # -----------------------
    # Health check
    # -----------------------
    async def health_check(self, sample_symbol: Optional[str] = None, timeframe: str = "1m", timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform a light health check:
        - client reachable
        - optionally fetch a recent ticker or a single candle to detect stale connections

        Returns dict: {"ok": bool, "reason": str, "last_seen": ts}
        """
        if not self._connected or not self._client:
            return {"ok": False, "reason": "not_connected", "last_seen": self._last_health}
        try:
            # check markets loaded
            markets = getattr(self._client, "markets", None)
            if markets is None:
                # try load_markets
                async def _load():
                    return await self._client.load_markets()
                await self._request_with_retry(_load, timeout=timeout)
            # optional quick fetch
            if sample_symbol:
                async def _one():
                    return await self._client.fetch_ohlcv(sample_symbol, timeframe, None, 1)
                res = await self._request_with_retry(_one, timeout=timeout)
                if not res:
                    return {"ok": False, "reason": "no_data", "last_seen": self._last_health}
            self._last_health = time.time()
            return {"ok": True, "reason": "healthy", "last_seen": self._last_health}
        except Exception as exc:
            return {"ok": False, "reason": str(exc), "last_seen": self._last_health}