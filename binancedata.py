#!/usr/bin/env python3
"""
binancedata.py — Binance Spot market scanner.

Fetches rolling 24-hour ticker statistics from Binance public market data,
filters USDT spot pairs, and reports:
  * Top gainers
  * Top losers
  * Highest USDT quote-volume pairs

Optionally exports the filtered market data to CSV.

Install:
    python -m pip install requests pandas

Examples:
    python binancedata.py
    python binancedata.py --top 30 --min-volume 5000000
    python binancedata.py --top 20 --csv binance_markets.csv
    python binancedata.py --quote BTC --top 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.binance.com"
TICKER_24HR_ENDPOINT = "/api/v3/ticker/24hr"
REQUEST_TIMEOUT_SECONDS = 20

NUMERIC_COLUMNS = (
    "lastPrice",
    "priceChange",
    "priceChangePercent",
    "weightedAvgPrice",
    "prevClosePrice",
    "lastQty",
    "bidPrice",
    "askPrice",
    "openPrice",
    "highPrice",
    "lowPrice",
    "volume",
    "quoteVolume",
    "openTime",
    "closeTime",
    "firstId",
    "lastId",
    "count",
)


class BinanceDataError(RuntimeError):
    """Raised when Binance market data cannot be retrieved or parsed."""


def fetch_24hr_tickers(session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch 24-hour ticker statistics for all Binance Spot symbols."""
    client = session or requests.Session()
    url = f"{BASE_URL}{TICKER_24HR_ENDPOINT}"

    try:
        response = client.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload: Any = response.json()
    except requests.Timeout as exc:
        raise BinanceDataError(
            f"Binance request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = ""
        if exc.response is not None:
            body = exc.response.text[:300]
        raise BinanceDataError(
            f"Binance returned HTTP {status}. {body}"
        ) from exc
    except requests.RequestException as exc:
        raise BinanceDataError(f"Network error while contacting Binance: {exc}") from exc
    except ValueError as exc:
        raise BinanceDataError("Binance returned invalid JSON.") from exc

    if not isinstance(payload, list) or not payload:
        raise BinanceDataError("Binance returned an empty or unexpected response.")

    df = pd.DataFrame(payload)
    required = {"symbol", "lastPrice", "priceChangePercent", "quoteVolume"}
    missing = required.difference(df.columns)
    if missing:
        raise BinanceDataError(
