"""
regime_classifier.py

Lightweight, deterministic regime classifier.

Contract:
- Accept historical metrics and market features
- Predict market regime: "trend" or "range"
- Return advisory dict (regime, suggested bias, confidence, explanation, features used)
- No execution or trading logic

Design:
- Default implementation is rule-based (no external ML dependency).
- Exposes sync and async predict methods.
- Minimal, auditable heuristics so downstream systems remain exchange-agnostic.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Advisory:
    regime: str               # "trend" or "range"
    bias: str                 # "long", "short", or "neutral"
    confidence: float         # 0.0 .. 1.0
    explanation: str
    features_used: Dict[str, Any]


class RegimeClassifier:
    """
    Simple regime classifier.

    Example usage:
        clf = RegimeClassifier()
        advisory = clf.predict(metrics=metrics_dict, features={"close": close_prices})
    """

    def __init__(
        self,
        min_samples: int = 20,
        trend_mean_ret_threshold: float = 0.0008,   # ~0.08% mean return per bar
        low_vol_threshold: float = 0.005,           # ~0.5% stdev per bar
    ):
        self.min_samples = min_samples
        self.trend_mean_ret_threshold = trend_mean_ret_threshold
        self.low_vol_threshold = low_vol_threshold

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _pct_returns_from_prices(prices: List[float]) -> List[float]:
        rets: List[float] = []
        for i in range(1, len(prices)):
            p0 = prices[i - 1]
            p1 = prices[i]
            if p0 == 0:
                rets.append(0.0)
            else:
                rets.append((p1 - p0) / p0)
        return rets

    @staticmethod
    def _safefloat(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _compute_basic_stats(self, prices: List[float]) -> Dict[str, Any]:
        rets = self._pct_returns_from_prices(prices)
        n = len(rets)
        mr = mean(rets) if n > 0 else 0.0
        vol = pstdev(rets) if n > 1 else 0.0
        # recent momentum: mean of last N returns
        recent_n = max(3, min(10, n))
        recent_mr = mean(rets[-recent_n:]) if n >= recent_n else mr
        return {"n": n, "mean_ret": mr, "vol": vol, "recent_mean_ret": recent_mr}

    # ----------------------
    # Prediction
    # ----------------------
    def predict(self, metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous prediction.

        Args:
            metrics: historical metrics (win rate, pnl, drawdown, optional) - advisory may use these to adjust confidence
            features: market features dictionary. Required key: "close" -> List[float] (most recent first->last or ascending)
                      Accepts other keys but only "close" is used by default.

        Returns:
            dict serializable Advisory-like object:
              {
                "regime": "trend"|"range",
                "bias": "long"|"short"|"neutral",
                "confidence": 0.0-1.0,
                "explanation": "...",
                "features_used": {...}
              }
        """
        metrics = metrics or {}
        features = features or {}

        prices = features.get("close") or features.get("prices") or features.get("price_series")
        if not prices or not isinstance(prices, (list, tuple)):
            raise ValueError("features must include a 'close' (price list) key with at least min_samples+1 values")

        prices = [self._safefloat(p) for p in prices]
        if len(prices) < self.min_samples:
            raise ValueError(f"need at least {self.min_samples} price points (got {len(prices)})")

        stats = self._compute_basic_stats(prices)
        n = stats["n"]
        mean_ret = stats["mean_ret"]
        vol = stats["vol"]
        recent_mean = stats["recent_mean_ret"]

        # Base heuristics:
        # - strong persistent mean return vs volatility => trend
        # - low volatility and small mean => range
        abs_mean = abs(mean_ret)
        regime = "range"
        bias = "neutral"
        confidence = 0.0
        explanation = ""

        # score components (non-negative)
        trend_score = max(0.0, (abs_mean - self.trend_mean_ret_threshold) / (self.trend_mean_ret_threshold if self.trend_mean_ret_threshold else 1.0))
        vol_score = max(0.0, (vol - self.low_vol_threshold) / (self.low_vol_threshold if self.low_vol_threshold else 1.0))

        # determine regime
        # if mean_ret noticeably larger than threshold and vol is reasonable -> trend
        if abs_mean >= self.trend_mean_ret_threshold and vol >= 1e-6:
            regime = "trend"
            # bias by sign of mean return (use recent_mean for sign robustness)
            bias = "long" if recent_mean > 0 else "short" if recent_mean < 0 else ("long" if mean_ret > 0 else "short")
            # confidence proportional to signal-to-noise (|mean| / vol) clipped
            snr = abs_mean / (vol + 1e-12)
            confidence = min(1.0, float(math.tanh(snr / 2.0)))  # squashed to [0,1]
            explanation = f"trend signal (mean_ret={mean_ret:.6f}, vol={vol:.6f}, snr={snr:.3f})"
        else:
            # range: low mean or insufficient SNR
            regime = "range"
            bias = "neutral"
            # confidence higher when volatility is low and mean close to zero
            vol_factor = max(0.0, 1.0 - (vol / (self.low_vol_threshold * 4.0 + 1e-12)))
            mean_factor = max(0.0, 1.0 - (abs_mean / (self.trend_mean_ret_threshold * 4.0 + 1e-12)))
            confidence = float(min(1.0, 0.25 + 0.75 * (vol_factor * mean_factor)))
            explanation = f"range signal (mean_ret={mean_ret:.6f}, vol={vol:.6f})"

        # modest adjustments from performance metrics (do not change regime, only nudge confidence)
        if metrics:
            win_rate = metrics.get("win_rate_pct") or metrics.get("win_rate") or metrics.get("wins")
            # if win rate is extreme, increase confidence slightly
            try:
                wr = float(win_rate) if win_rate is not None else None
            except Exception:
                wr = None
            if wr is not None:
                # normalize wr from [0,100] to [-1,1] then scale
                wr_norm = (wr / 50.0) - 1.0
                confidence = float(max(0.0, min(1.0, confidence + 0.05 * wr_norm)))

        # ensure bounds
        confidence = max(0.0, min(1.0, confidence))

        advisory = Advisory(
            regime=regime,
            bias=bias,
            confidence=confidence,
            explanation=explanation,
            features_used={"n_rets": n, "mean_ret": mean_ret, "vol": vol, "recent_mean": recent_mean},
        )
        # Return plain dict for easy serialization
        return {
            "regime": advisory.regime,
            "bias": advisory.bias,
            "confidence": round(advisory.confidence, 4),
            "explanation": advisory.explanation,
            "features_used": advisory.features_used,
        }

    async def predict_async(self, metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async wrapper (runs predict in default loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, metrics, features)


# Module-level convenience
_default_clf = RegimeClassifier()


def predict_regime(metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function using the default classifier instance."""
    return _default_clf.predict(metrics=metrics, features=features)


async def predict_regime_async(metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return await _default_clf.predict_async(metrics=metrics, features=features)


# Example (local) usage
if __name__ == "__main__":
    # tiny demo with synthetic prices
    import random
    base = 100.0
    # generate gentle uptrend
    prices = [base + i * 0.02 + random.gauss(0, 0.2) for i in range(200)]
    adv = predict_regime({}, {"close": prices})
    print("Advisory:", adv)
```# filepath: /home/oladan/Trading-Platform/ai/regime_classifier.py
"""
regime_classifier.py

Lightweight, deterministic regime classifier.

Contract:
- Accept historical metrics and market features
- Predict market regime: "trend" or "range"
- Return advisory dict (regime, suggested bias, confidence, explanation, features used)
- No execution or trading logic

Design:
- Default implementation is rule-based (no external ML dependency).
- Exposes sync and async predict methods.
- Minimal, auditable heuristics so downstream systems remain exchange-agnostic.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Advisory:
    regime: str               # "trend" or "range"
    bias: str                 # "long", "short", or "neutral"
    confidence: float         # 0.0 .. 1.0
    explanation: str
    features_used: Dict[str, Any]


class RegimeClassifier:
    """
    Simple regime classifier.

    Example usage:
        clf = RegimeClassifier()
        advisory = clf.predict(metrics=metrics_dict, features={"close": close_prices})
    """

    def __init__(
        self,
        min_samples: int = 20,
        trend_mean_ret_threshold: float = 0.0008,   # ~0.08% mean return per bar
        low_vol_threshold: float = 0.005,           # ~0.5% stdev per bar
    ):
        self.min_samples = min_samples
        self.trend_mean_ret_threshold = trend_mean_ret_threshold
        self.low_vol_threshold = low_vol_threshold

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _pct_returns_from_prices(prices: List[float]) -> List[float]:
        rets: List[float] = []
        for i in range(1, len(prices)):
            p0 = prices[i - 1]
            p1 = prices[i]
            if p0 == 0:
                rets.append(0.0)
            else:
                rets.append((p1 - p0) / p0)
        return rets

    @staticmethod
    def _safefloat(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _compute_basic_stats(self, prices: List[float]) -> Dict[str, Any]:
        rets = self._pct_returns_from_prices(prices)
        n = len(rets)
        mr = mean(rets) if n > 0 else 0.0
        vol = pstdev(rets) if n > 1 else 0.0
        # recent momentum: mean of last N returns
        recent_n = max(3, min(10, n))
        recent_mr = mean(rets[-recent_n:]) if n >= recent_n else mr
        return {"n": n, "mean_ret": mr, "vol": vol, "recent_mean_ret": recent_mr}

    # ----------------------
    # Prediction
    # ----------------------
    def predict(self, metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous prediction.

        Args:
            metrics: historical metrics (win rate, pnl, drawdown, optional) - advisory may use these to adjust confidence
            features: market features dictionary. Required key: "close" -> List[float] (most recent first->last or ascending)
                      Accepts other keys but only "close" is used by default.

        Returns:
            dict serializable Advisory-like object:
              {
                "regime": "trend"|"range",
                "bias": "long"|"short"|"neutral",
                "confidence": 0.0-1.0,
                "explanation": "...",
                "features_used": {...}
              }
        """
        metrics = metrics or {}
        features = features or {}

        prices = features.get("close") or features.get("prices") or features.get("price_series")
        if not prices or not isinstance(prices, (list, tuple)):
            raise ValueError("features must include a 'close' (price list) key with at least min_samples+1 values")

        prices = [self._safefloat(p) for p in prices]
        if len(prices) < self.min_samples:
            raise ValueError(f"need at least {self.min_samples} price points (got {len(prices)})")

        stats = self._compute_basic_stats(prices)
        n = stats["n"]
        mean_ret = stats["mean_ret"]
        vol = stats["vol"]
        recent_mean = stats["recent_mean_ret"]

        # Base heuristics:
        # - strong persistent mean return vs volatility => trend
        # - low volatility and small mean => range
        abs_mean = abs(mean_ret)
        regime = "range"
        bias = "neutral"
        confidence = 0.0
        explanation = ""

        # score components (non-negative)
        trend_score = max(0.0, (abs_mean - self.trend_mean_ret_threshold) / (self.trend_mean_ret_threshold if self.trend_mean_ret_threshold else 1.0))
        vol_score = max(0.0, (vol - self.low_vol_threshold) / (self.low_vol_threshold if self.low_vol_threshold else 1.0))

        # determine regime
        # if mean_ret noticeably larger than threshold and vol is reasonable -> trend
        if abs_mean >= self.trend_mean_ret_threshold and vol >= 1e-6:
            regime = "trend"
            # bias by sign of mean return (use recent_mean for sign robustness)
            bias = "long" if recent_mean > 0 else "short" if recent_mean < 0 else ("long" if mean_ret > 0 else "short")
            # confidence proportional to signal-to-noise (|mean| / vol) clipped
            snr = abs_mean / (vol + 1e-12)
            confidence = min(1.0, float(math.tanh(snr / 2.0)))  # squashed to [0,1]
            explanation = f"trend signal (mean_ret={mean_ret:.6f}, vol={vol:.6f}, snr={snr:.3f})"
        else:
            # range: low mean or insufficient SNR
            regime = "range"
            bias = "neutral"
            # confidence higher when volatility is low and mean close to zero
            vol_factor = max(0.0, 1.0 - (vol / (self.low_vol_threshold * 4.0 + 1e-12)))
            mean_factor = max(0.0, 1.0 - (abs_mean / (self.trend_mean_ret_threshold * 4.0 + 1e-12)))
            confidence = float(min(1.0, 0.25 + 0.75 * (vol_factor * mean_factor)))
            explanation = f"range signal (mean_ret={mean_ret:.6f}, vol={vol:.6f})"

        # modest adjustments from performance metrics (do not change regime, only nudge confidence)
        if metrics:
            win_rate = metrics.get("win_rate_pct") or metrics.get("win_rate") or metrics.get("wins")
            # if win rate is extreme, increase confidence slightly
            try:
                wr = float(win_rate) if win_rate is not None else None
            except Exception:
                wr = None
            if wr is not None:
                # normalize wr from [0,100] to [-1,1] then scale
                wr_norm = (wr / 50.0) - 1.0
                confidence = float(max(0.0, min(1.0, confidence + 0.05 * wr_norm)))

        # ensure bounds
        confidence = max(0.0, min(1.0, confidence))

        advisory = Advisory(
            regime=regime,
            bias=bias,
            confidence=confidence,
            explanation=explanation,
            features_used={"n_rets": n, "mean_ret": mean_ret, "vol": vol, "recent_mean": recent_mean},
        )
        # Return plain dict for easy serialization
        return {
            "regime": advisory.regime,
            "bias": advisory.bias,
            "confidence": round(advisory.confidence, 4),
            "explanation": advisory.explanation,
            "features_used": advisory.features_used,
        }

    async def predict_async(self, metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async wrapper (runs predict in default loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, metrics, features)


# Module-level convenience
_default_clf = RegimeClassifier()


def predict_regime(metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function using the default classifier instance."""
    return _default_clf.predict(metrics=metrics, features=features)


async def predict_regime_async(metrics: Optional[Dict[str, Any]] = None, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return await _default_clf.predict_async(metrics=metrics, features=features)


# Example (local) usage
if __name__ == "__main__":
    # tiny demo with synthetic prices
    import random
    base = 100.0
    # generate gentle uptrend
    prices = [base + i * 0.02 + random.gauss(0, 0.2) for i in range(200)]
    adv = predict_regime({}, {"close": prices})
    print("Advisory:", adv)