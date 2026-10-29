#!/usr/bin/env python3
"""
binancedata.py — Binance market scanner.

Fetches rolling 24-hour ticker statistics from Binance market data, filters
pairs by quote asset, and reports:
  * Top gainers
  * Top losers
  * Highest quote-volume pairs

Public market data needs no API key. Private account data (``--account``)
requires a Binance API key/secret for the selected environment:





   
Demo trading keys are created at <https://demo.binance.com> (they are separate
from live keys). Live keys are created at
<https://www.binance.com/en/my/settings/api-management>.

Install:
    python -m pip install requests pandas python-dotenv

Examples:
    python binancedata.py
    python binancedata.py --top 30 --min-volume 5000000
    python binancedata.py --market spot --env live --top 20
    python binancedata.py --top 20 --csv binance_markets.csv
    python binancedata.py --quote BTC --top 10
    python binancedata.py --account
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

REQUEST_TIMEOUT_SECONDS = 20
RECV_WINDOW_MS = 5000

BASE_URLS = {
    ("spot", "live"): "https://api.binance.com",
    ("spot", "demo"): "https://demo-api.binance.com",
    ("futures", "live"): "https://fapi.binance.com",
    ("futures", "demo"): "https://demo-fapi.binance.com",
}

TICKER_24HR_PATHS = {
    "spot": "/api/v3/ticker/24hr",
    "futures": "/fapi/v1/ticker/24hr",
}

EXCHANGE_INFO_PATHS = {
    "spot": "/api/v3/exchangeInfo",
    "futures": "/fapi/v1/exchangeInfo",
}

ACCOUNT_PATHS = {
    "spot": "/api/v3/account",
    "futures": "/fapi/v2/balance",
}

DEFAULT_MARKET = os.environ.get("BINANCE_MARKET", "futures")
DEFAULT_ENVIRONMENT = os.environ.get("BINANCE_ENV", "demo")

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

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

DISPLAY_COLUMNS = (
    "symbol",
    "lastPrice",
    "priceChangePercent",
    "quoteVolume",
    "highPrice",
    "lowPrice",
)


class BinanceDataError(RuntimeError):
    """Raised when Binance market data cannot be retrieved or parsed."""


def resolve_base_url(market: str, environment: str, override: str | None) -> str:
    """Return the REST base URL for a market/environment pair."""
    if override:
        return override.rstrip("/")
    try:
        return BASE_URLS[(market, environment)]
    except KeyError as exc:
        raise BinanceDataError(
            f"Unsupported market/environment: {market}/{environment}"
        ) from exc


def _request_json(
    client: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET a Binance endpoint and decode JSON with helpful error messages."""
    try:
        response = client.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise BinanceDataError(
            f"Binance request timed out after {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:300] if exc.response is not None else ""
        raise BinanceDataError(f"Binance returned HTTP {status}. {body}") from exc
    except requests.RequestException as exc:
        raise BinanceDataError(f"Network error while contacting Binance: {exc}") from exc
    except ValueError as exc:
        raise BinanceDataError("Binance returned invalid JSON.") from exc


def fetch_24hr_tickers(
    base_url: str,
    market: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch 24-hour ticker statistics for all symbols on a market."""
    client = session or requests.Session()
    url = f"{base_url}{TICKER_24HR_PATHS[market]}"
    payload = _request_json(client, url)

    if not isinstance(payload, list) or not payload:
        raise BinanceDataError("Binance returned an empty or unexpected response.")

    df = pd.DataFrame(payload)
    required = {"symbol", "lastPrice", "priceChangePercent", "quoteVolume"}
    missing = required.difference(df.columns)
    if missing:
        raise BinanceDataError(
            "Binance ticker response is missing columns: " + ", ".join(sorted(missing))
        )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def fetch_tradable_symbols(
    base_url: str,
    market: str,
    quote: str,
    session: requests.Session | None = None,
) -> set[str]:
    """Return symbols currently trading and quoted in ``quote``."""
    client = session or requests.Session()
    url = f"{base_url}{EXCHANGE_INFO_PATHS[market]}"
    payload = _request_json(client, url)

    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(symbols, list):
        raise BinanceDataError("Binance exchangeInfo response is missing symbols.")

    tradable: set[str] = set()
    for item in symbols:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != quote:
            continue
        symbol = item.get("symbol")
        if not symbol or "_" in symbol:
            continue
        if market == "spot" and _is_leveraged_token(symbol, quote):
            continue
        tradable.add(symbol)

    if not tradable:
        raise BinanceDataError(f"No trading symbols found for quote asset {quote}.")

    return tradable


def _is_leveraged_token(symbol: str, quote: str) -> bool:
    base = symbol[: -len(quote)] if symbol.endswith(quote) else symbol
    return any(base.endswith(suffix) for suffix in LEVERAGED_SUFFIXES)


def build_snapshot(
    base_url: str,
    market: str,
    quote: str,
    min_volume: float,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch tickers and keep only liquid, tradable pairs for ``quote``."""
    client = session or requests.Session()
    tickers = fetch_24hr_tickers(base_url, market, client)
    tradable = fetch_tradable_symbols(base_url, market, quote, client)

    df = tickers[tickers["symbol"].isin(tradable)].copy()
    df = df[df["quoteVolume"] >= min_volume]
    df = df.dropna(subset=["lastPrice", "priceChangePercent", "quoteVolume"])

    if df.empty:
        raise BinanceDataError(
            f"No {quote} pairs matched the filters "
            f"(min-volume={min_volume:,.0f})."
        )

    return df


def _signed_query(api_secret: str, params: dict[str, Any]) -> str:
    query = urlencode(params)
    signature = hmac.new(
        api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{query}&signature={signature}"


def fetch_account_balance(
    base_url: str,
    market: str,
    api_key: str,
    api_secret: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch non-zero account balances using a signed request."""
    client = session or requests.Session()
    params = {"timestamp": int(time.time() * 1000), "recvWindow": RECV_WINDOW_MS}
    url = f"{base_url}{ACCOUNT_PATHS[market]}?{_signed_query(api_secret, params)}"

    try:
        payload = _request_json(client, url, headers={"X-MBX-APIKEY": api_key})
    except BinanceDataError as exc:
        raise BinanceDataError(
            f"{exc} (check BINANCE_API_KEY/BINANCE_API_SECRET and that the keys "
            f"belong to the selected {market} environment)"
        ) from exc

    if market == "spot":
        balances = payload.get("balances", []) if isinstance(payload, dict) else []
        rows = [
            {"asset": item.get("asset"), "free": item.get("free"), "locked": item.get("locked")}
            for item in balances
        ]
    else:
        rows = [
            {"asset": item.get("asset"), "balance": item.get("balance")}
            for item in payload
            if isinstance(item, dict)
        ] if isinstance(payload, list) else []

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for column in ("free", "locked", "balance"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    value_columns = [c for c in ("free", "locked", "balance") if c in df.columns]
    return df[(df[value_columns] != 0).any(axis=1)].reset_index(drop=True)


def _print_table(df: pd.DataFrame, title: str, top: int) -> None:
    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    table = df[columns].head(top).copy()
    table = table.rename(
        columns={
            "lastPrice": "price",
            "priceChangePercent": "change_%",
            "quoteVolume": "volume",
            "highPrice": "high",
            "lowPrice": "low",
        }
    )
    print(f"\n{title}")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(table.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))


def report(df: pd.DataFrame, top: int, quote: str, min_volume: float) -> None:
    """Print gainers, losers and volume leaders."""
    liquid = df[df["quoteVolume"] >= min_volume]
    print(
        f"\n{len(df)} {quote} pairs | min 24h volume {min_volume:,.0f} "
        f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"
    )
    _print_table(
        liquid.sort_values("priceChangePercent", ascending=False), "Top gainers", top
    )
    _print_table(
        liquid.sort_values("priceChangePercent", ascending=True), "Top losers", top
    )
    _print_table(
        liquid.sort_values("quoteVolume", ascending=False), "Top volume", top
    )


def export_csv(df: pd.DataFrame, path: Path) -> None:
    columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    df.sort_values("quoteVolume", ascending=False)[columns].to_csv(
        path, index=False
    )
    print(f"\nWrote {len(df)} rows to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Binance 24h tickers for gainers, losers and volume."
    )
    parser.add_argument("--top", type=int, default=20, help="rows per table (default: 20)")
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1_000_000,
        help="minimum 24h quote volume (default: 1000000)",
    )
    parser.add_argument("--quote", default="USDT", help="quote asset (default: USDT)")
    parser.add_argument(
        "--market",
        choices=("spot", "futures"),
        default=DEFAULT_MARKET,
        help=f"market type (default: {DEFAULT_MARKET})",
    )
    parser.add_argument(
        "--env",
        dest="environment",
        choices=("demo", "live"),
        default=DEFAULT_ENVIRONMENT,
        help=f"Binance environment (default: {DEFAULT_ENVIRONMENT})",
    )
    parser.add_argument("--base-url", default=None, help="override the REST base URL")
    parser.add_argument("--csv", type=Path, default=None, help="export filtered data to CSV")
    parser.add_argument(
        "--account",
        action="store_true",
        help="also print account balances (requires API key/secret)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.top < 1:
        print("error: --top must be at least 1", file=sys.stderr)
        return 2
    if args.min_volume < 0:
        print("error: --min-volume must be >= 0", file=sys.stderr)
        return 2

    quote = args.quote.upper()
    base_url = resolve_base_url(args.market, args.environment, args.base_url)
    print(f"Binance {args.market} / {args.environment} -> {base_url}")

    try:
        with requests.Session() as session:
            df = build_snapshot(
                base_url, args.market, quote, args.min_volume, session
            )
            report(df, args.top, quote, args.min_volume)

            if args.csv:
                export_csv(df, args.csv)

            if args.account:
                api_key = os.environ.get("BINANCE_API_KEY")
                api_secret = os.environ.get("BINANCE_API_SECRET")
                if not api_key or not api_secret:
                    raise BinanceDataError(
                        "BINANCE_API_KEY and BINANCE_API_SECRET must be set for --account."
                    )
                balances = fetch_account_balance(
                    base_url, args.market, api_key, api_secret, session
                )
                print("\nAccount balances")
                if balances.empty:
                    print("No non-zero balances.")
                else:
                    print(balances.to_string(index=False))
    except BinanceDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
