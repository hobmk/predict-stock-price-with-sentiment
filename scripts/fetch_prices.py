# 지정 10종목, 최근 5년 일봉 OHLCV 수집

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

TICKERS = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "207940.KS",  # 삼성바이오로직스
    "012450.KS",  # 한화에어로스페이스
    "005380.KS",  # 현대차
    "000270.KS",  # 기아
    "105560.KS",  # KB금융
    "034020.KS",  # 두산에너빌리티
    "068270.KS",  # 셀트리온
    "035420.KS",  # 네이버
]


def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)


def fetch_one(ticker: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = (
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .reset_index()
        .rename(columns={"Date": "date"})
    )
    df.insert(1, "ticker", ticker)
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def main():
    ensure_dirs()
    end_dt = datetime.today().date()
    start_dt = end_dt - timedelta(days=365 * 5 + 2)  # 5년 + 버퍼
    start, end = start_dt.isoformat(), end_dt.isoformat()

    all_rows = []
    for i, tkr in enumerate(TICKERS, 1):
        print(f"[{i:02d}/10] {tkr} {start}~{end}")
        try:
            df = fetch_one(tkr, start, end, "1d")
            if df.empty:
                print("  -> empty")
                continue
            out = f"data/raw/prices_{tkr}.csv"
            df.to_csv(out, index=False)
            all_rows.append(df)
            print(f"  -> saved {len(df)} rows to {out}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  -> FAIL: {e}")

    if all_rows:
        merged = pd.concat(all_rows, ignore_index=True).sort_values(["ticker", "date"])
        merged.to_csv("data/raw/prices.csv", index=False)
        print(
            f"[MERGED] data/raw/prices.csv ({len(merged)} rows, {merged['ticker'].nunique()} tickers)"
        )


if __name__ == "__main__":
    main()
