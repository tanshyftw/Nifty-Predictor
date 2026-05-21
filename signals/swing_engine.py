"""Rule-based swing trade idea engine.

The goal is not to forecast a price. It is to surface repeatable setups with
transparent entry, stop, target, and invalidation logic so the dashboard can
behave like a self-contained action system.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingIdea:
    symbol: str
    company: str
    sector: str
    setup: str
    direction: str
    score: float
    close: float
    entry: float
    stop: float
    target_1: float
    target_2: float
    reward_risk: float
    horizon: str
    rationale: str
    invalidation: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_swing_ideas(
    histories: dict[str, pd.DataFrame],
    metadata: dict[str, tuple[str, str]],
    *,
    index_close: pd.Series | None = None,
    min_score: float = 60.0,
) -> pd.DataFrame:
    """Compute ranked swing ideas from OHLCV history."""
    ideas: list[SwingIdea] = []
    index_ret_20 = _return_pct(index_close, 20) if index_close is not None else np.nan

    for symbol, df in histories.items():
        if df is None or df.empty:
            continue
        idea = score_symbol(
            symbol,
            df,
            company=metadata.get(symbol, (symbol, "Unknown"))[0],
            sector=metadata.get(symbol, (symbol, "Unknown"))[1],
            index_ret_20=index_ret_20,
        )
        if idea and idea.score >= min_score:
            ideas.append(idea)

    if not ideas:
        return pd.DataFrame()
    out = pd.DataFrame([idea.as_dict() for idea in ideas])
    return out.sort_values(["score", "reward_risk"], ascending=False).reset_index(drop=True)


def score_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    company: str,
    sector: str,
    index_ret_20: float = np.nan,
) -> SwingIdea | None:
    """Return the best swing setup for one symbol, if any."""
    work = _prepare_history(df)
    if work.empty or len(work) < 80:
        return None

    close = work["Close"]
    high = work["High"]
    low = work["Low"]
    volume = work["Volume"] if "Volume" in work.columns else pd.Series(index=work.index, dtype=float)

    last = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else ema50
    rsi = _rsi(close)
    atr = _atr(work).iloc[-1]
    atr = float(atr) if np.isfinite(atr) and atr > 0 else max(last * 0.025, 1.0)
    ret_5 = _return_pct(close, 5)
    ret_20 = _return_pct(close, 20)
    rel_20 = ret_20 - index_ret_20 if np.isfinite(index_ret_20) else 0.0
    vol_ratio = _volume_ratio(volume)
    prev_20_high = float(high.shift(1).rolling(20).max().iloc[-1])
    prev_20_low = float(low.shift(1).rolling(20).min().iloc[-1])
    support = float(low.tail(20).min())

    candidates: list[SwingIdea] = []

    breakout_score = 0.0
    bits = []
    if last > prev_20_high:
        breakout_score += 30
        bits.append("20-day breakout")
    if last > ema20 > ema50:
        breakout_score += 20
        bits.append("EMA stack bullish")
    if 50 <= rsi <= 72:
        breakout_score += 15
        bits.append(f"RSI {rsi:.0f} with room")
    if vol_ratio >= 1.15:
        breakout_score += 15
        bits.append(f"volume {vol_ratio:.1f}x")
    if rel_20 > 0:
        breakout_score += 10
        bits.append(f"relative strength {rel_20:+.1f} pp")
    if ret_5 > 0:
        breakout_score += 10
    candidates.append(_make_idea(
        symbol, company, sector, "Momentum breakout", "LONG", breakout_score,
        last, max(last, prev_20_high), max(support, last - 1.5 * atr), atr,
        bits or ["breakout conditions forming"], "2-4 weeks",
    ))

    pullback_score = 0.0
    bits = []
    distance_ema20 = (last - ema20) / ema20 * 100 if ema20 else 0.0
    if last > ema50 > ema200:
        pullback_score += 25
        bits.append("primary trend up")
    if -2.5 <= distance_ema20 <= 1.5:
        pullback_score += 25
        bits.append("near EMA-20")
    if 38 <= rsi <= 58:
        pullback_score += 15
        bits.append(f"RSI reset {rsi:.0f}")
    if ret_20 > 3 and ret_5 <= 1.5:
        pullback_score += 15
        bits.append("recent pause after strength")
    if rel_20 > 0:
        pullback_score += 10
        bits.append("outperforming index")
    if last > support * 1.01:
        pullback_score += 10
    candidates.append(_make_idea(
        symbol, company, sector, "Trend pullback", "LONG", pullback_score,
        last, last, min(support, ema50), atr, bits or ["pullback conditions forming"], "1-3 weeks",
    ))

    reversion_score = 0.0
    bits = []
    if rsi <= 35:
        reversion_score += 30
        bits.append(f"RSI oversold {rsi:.0f}")
    if last <= prev_20_low * 1.03:
        reversion_score += 20
        bits.append("near 20-day low")
    if last >= ema200 * 0.95:
        reversion_score += 15
        bits.append("not deeply broken vs long trend")
    if vol_ratio < 1.8:
        reversion_score += 10
        bits.append("selloff not volume-panic")
    if ret_5 < -3:
        reversion_score += 15
    if rel_20 > -5:
        reversion_score += 10
    candidates.append(_make_idea(
        symbol, company, sector, "Mean-reversion bounce", "LONG", reversion_score,
        last, last, last - 1.2 * atr, atr, bits or ["mean-reversion conditions forming"], "3-10 sessions",
    ))

    candidates = [idea for idea in candidates if idea is not None and idea.reward_risk >= 1.2]
    if not candidates:
        return None
    return max(candidates, key=lambda idea: (idea.score, idea.reward_risk))


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [c[0] if isinstance(c, tuple) else c for c in work.columns]
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(work.columns)):
        return pd.DataFrame()
    return work.dropna(subset=["Close", "High", "Low"])


def _make_idea(
    symbol: str,
    company: str,
    sector: str,
    setup: str,
    direction: str,
    score: float,
    close: float,
    entry: float,
    stop: float,
    atr: float,
    rationale_bits: list[str],
    horizon: str,
) -> SwingIdea | None:
    risk = max(entry - stop, atr * 0.6)
    if risk <= 0 or close <= 0:
        return None
    target_1 = entry + 1.5 * risk
    target_2 = entry + 2.5 * risk
    reward_risk = (target_1 - entry) / risk if risk else 0.0
    return SwingIdea(
        symbol=symbol,
        company=company,
        sector=sector,
        setup=setup,
        direction=direction,
        score=round(float(score), 1),
        close=round(float(close), 2),
        entry=round(float(entry), 2),
        stop=round(float(stop), 2),
        target_1=round(float(target_1), 2),
        target_2=round(float(target_2), 2),
        reward_risk=round(float(reward_risk), 2),
        horizon=horizon,
        rationale="; ".join(rationale_bits),
        invalidation=f"Close below {stop:.2f} or setup score drops below 50",
    )


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    return val if np.isfinite(val) else 50.0


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _return_pct(series: pd.Series | None, days: int) -> float:
    if series is None or len(series.dropna()) <= days:
        return np.nan
    s = series.dropna()
    base = float(s.iloc[-days - 1])
    last = float(s.iloc[-1])
    return ((last / base) - 1) * 100 if base else np.nan


def _volume_ratio(volume: pd.Series) -> float:
    if volume is None or volume.dropna().empty or len(volume.dropna()) < 21:
        return 1.0
    v = volume.dropna()
    avg = float(v.shift(1).tail(20).mean())
    last = float(v.iloc[-1])
    return last / avg if avg else 1.0
