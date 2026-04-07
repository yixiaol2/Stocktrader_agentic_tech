"""
Market Data Component
Fetches real stock data using yfinance and computes derived features.
No LLM calls are made in this module.
"""

import time
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


def fetch_all_market_data(tickers: list, max_retries: int = 5) -> dict:
    """
    Batch-fetch historical price data for all tickers at once (fewer API calls).
    Returns a dict mapping ticker -> market data dict.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    raw = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                group_by="ticker",
            )
            if raw is not None and not raw.empty:
                break
        except Exception:
            pass
        if attempt < max_retries - 1:
            wait = 15 * (attempt + 1)
            print(f"  Rate limited, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    if raw is None or raw.empty:
        raise ValueError("Failed to fetch data for any tickers after retries")

    results = {}
    for ticker in tickers:
        hist = raw[ticker].dropna()
        results[ticker] = _compute_indicators(ticker, hist)

    return results


def fetch_market_data(ticker: str, max_retries: int = 5) -> dict:
    """
    Fetch historical price data and compute indicators for a single stock ticker.
    """
    result = fetch_all_market_data([ticker], max_retries=max_retries)
    return result[ticker]


def _compute_indicators(ticker: str, hist) -> dict:
    """Compute all derived features from historical data."""
    if len(hist) < 50:
        raise ValueError(f"Not enough data for {ticker}: only {len(hist)} rows")

    # Basic price data
    current_price = round(float(hist["Close"].iloc[-1]), 2)
    price_30d_ago = round(float(hist["Close"].iloc[-30]), 2) if len(hist) >= 30 else current_price
    pct_change_30d = round(((current_price - price_30d_ago) / price_30d_ago) * 100, 2)

    # --- Momentum Trader features ---
    moving_avg_20d = round(float(hist["Close"].tail(20).mean()), 2)
    moving_avg_50d = round(float(hist["Close"].tail(50).mean()), 2)

    avg_volume_10d = int(hist["Volume"].tail(10).mean())
    avg_volume_30d = int(hist["Volume"].tail(30).mean())
    volume_trend = "increasing" if avg_volume_10d > avg_volume_30d else "decreasing"

    last_30_close = hist["Close"].tail(30)
    daily_returns_30d = last_30_close.pct_change().dropna()
    positive_days = int((daily_returns_30d > 0).sum())
    negative_days = int((daily_returns_30d < 0).sum())
    avg_daily_return = round(float(daily_returns_30d.mean()) * 100, 4)

    # --- Value Contrarian features ---
    high_52w = round(float(hist["High"].max()), 2)
    low_52w = round(float(hist["Low"].min()), 2)
    distance_from_52w_high = round(((current_price - high_52w) / high_52w) * 100, 2)
    distance_from_52w_low = round(((current_price - low_52w) / low_52w) * 100, 2)

    last_90_close = hist["Close"].tail(90)
    rolling_max = last_90_close.cummax()
    drawdown = ((last_90_close - rolling_max) / rolling_max) * 100
    max_drawdown_90d = round(float(drawdown.min()), 2)

    # RSI (14-day)
    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = float(gain.tail(15).head(14).mean())
    avg_loss = float(loss.tail(15).head(14).mean())
    gains_14 = gain.tail(14)
    losses_14 = loss.tail(14)
    for i in range(len(gains_14)):
        avg_gain = (avg_gain * 13 + float(gains_14.iloc[i])) / 14
        avg_loss = (avg_loss * 13 + float(losses_14.iloc[i])) / 14
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)), 2)

    # --- Shared features ---
    volatility_30d = round(float(daily_returns_30d.std()), 4)
    avg_daily_volume = int(hist["Volume"].tail(30).mean())

    return {
        "ticker": ticker,
        "current_price": current_price,
        "price_30d_ago": price_30d_ago,
        "pct_change_30d": pct_change_30d,
        "moving_avg_20d": moving_avg_20d,
        "moving_avg_50d": moving_avg_50d,
        "avg_volume_10d": avg_volume_10d,
        "avg_volume_30d": avg_volume_30d,
        "volume_trend": volume_trend,
        "positive_days_30d": positive_days,
        "negative_days_30d": negative_days,
        "avg_daily_return_pct": avg_daily_return,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "distance_from_52w_high_pct": distance_from_52w_high,
        "distance_from_52w_low_pct": distance_from_52w_low,
        "max_drawdown_90d_pct": max_drawdown_90d,
        "rsi_14d": rsi,
        "volatility_30d": volatility_30d,
        "avg_daily_volume": avg_daily_volume,
    }


def format_market_data_for_llm(data: dict) -> str:
    """Format market data dictionary into a readable string for LLM consumption."""
    return f"""Stock: {data['ticker']}
Current Price: ${data['current_price']}
Price 30 Days Ago: ${data['price_30d_ago']}
30-Day Price Change: {data['pct_change_30d']}%

Moving Averages:
  20-Day MA: ${data['moving_avg_20d']}
  50-Day MA: ${data['moving_avg_50d']}
  20-Day MA vs 50-Day MA: {"20d ABOVE 50d (bullish)" if data['moving_avg_20d'] > data['moving_avg_50d'] else "20d BELOW 50d (bearish)"}

Volume:
  Average Volume (last 10 days): {data['avg_volume_10d']:,}
  Average Volume (last 30 days): {data['avg_volume_30d']:,}
  Volume Trend: {data['volume_trend']}

Daily Returns (last 30 days):
  Positive Days: {data['positive_days_30d']}
  Negative Days: {data['negative_days_30d']}
  Average Daily Return: {data['avg_daily_return_pct']}%

52-Week Range:
  52-Week High: ${data['high_52w']}
  52-Week Low: ${data['low_52w']}
  Distance from 52-Week High: {data['distance_from_52w_high_pct']}%
  Distance from 52-Week Low: {data['distance_from_52w_low_pct']}%

Risk Indicators:
  Max Drawdown (90 days): {data['max_drawdown_90d_pct']}%
  RSI (14-day): {data['rsi_14d']}
  Volatility (30-day daily return std): {data['volatility_30d']}"""


def get_market_data_summary(data: dict) -> dict:
    """Extract the summary fields for JSON output."""
    return {
        "current_price": data["current_price"],
        "price_30d_ago": data["price_30d_ago"],
        "pct_change_30d": data["pct_change_30d"],
        "avg_daily_volume": data["avg_daily_volume"],
        "volatility_30d": data["volatility_30d"],
        "moving_avg_20d": data["moving_avg_20d"],
        "moving_avg_50d": data["moving_avg_50d"],
        "high_52w": data["high_52w"],
        "low_52w": data["low_52w"],
        "distance_from_52w_high_pct": data["distance_from_52w_high_pct"],
        "rsi_14d": data["rsi_14d"],
        "max_drawdown_90d_pct": data["max_drawdown_90d_pct"],
    }
