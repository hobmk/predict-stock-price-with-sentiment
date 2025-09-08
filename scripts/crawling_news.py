# scripts/crawling_news.py
# 네이버 금융 주요뉴스 → 날짜별/페이지별 크롤링 → 키워드 매칭 → 종목별 저장
# 휴장일/주말 기사는 "다음 거래일"로 올려붙임(15:20 이후 기사도 다음 거래일)
#
# 사용:
#   pip install requests beautifulsoup4 pandas
#   python -u scripts/crawling_news.py --prices data/raw/prices.csv
#   # 또는
#   python -u scripts/crawling_news.py --start 2020-09-03 --end 2025-09-03

import os
import re
import html
import time
import argparse
import bisect
from typing import List, Dict, Optional
from urllib.parse import urljoin
from datetime import datetime, timedelta, time as dtime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# 기본 설정
KST = timezone(timedelta(hours=9))
MAINNEWS_URL = "https://finance.naver.com/news/mainnews.naver"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": MAINNEWS_URL,
}

MAX_PAGES_PER_DAY = 10
REQUEST_SLEEP = 0.10
MARKET_CLOSE = dtime(15, 20)

OUT_DIR = "data/news_major"
DEBUG_DIR = "debug"

# 대상 티커(야후 형식) 및 네이버 코드
TICKERS = [
    "005930.KS","000660.KS","207940.KS","012450.KS","005380.KS",
    "000270.KS","105560.KS","034020.KS","068270.KS","035420.KS",
]
TICKER_TO_CODE = {
    "005930.KS":"005930","000660.KS":"000660","207940.KS":"207940","012450.KS":"012450",
    "005380.KS":"005380","000270.KS":"000270","105560.KS":"105560","034020.KS":"034020",
    "068270.KS":"068270","035420.KS":"035420",
}

# 회사/브랜드/약칭 키워드(정규식)
KEYWORDS: Dict[str, List[str]] = {
    "005930.KS": [r"삼성전자", r"\bGalaxy\b", r"갤럭시", r"반도체"],
    "000660.KS": [r"SK하이닉스", r"하이닉스", r"\bHBM\b", r"메모리"],
    "207940.KS": [r"삼성바이오로직스", r"삼바", r"\bCMO\b|\bCDMO\b"],
    "012450.KS": [r"한화에어로스페이스", r"\b한화\b", r"\bK9\b", r"누리호", r"방산"],
    "005380.KS": [r"현대차|현대자동차", r"\bIONIQ\b|아이오닉", r"\bEV\b|전기차"],
    "000270.KS": [r"기아|\bKIA\b", r"스포티지", r"EV\d+"],
    "105560.KS": [r"KB금융(그룹)?", r"KB국민은행", r"KB증권"],
    "034020.KS": [r"두산에너빌리티|두산중공업", r"원전|터빈|수소"],
    "068270.KS": [r"셀트리온", r"램시마", r"바이오시밀러"],
    "035420.KS": [r"네이버|\bNAVER\b", r"라인(야후)?", r"검색광고|커머스"],
}

# 광범위 토큰 보정
CO_OCCUR = {
    r"삼성": [r"전자|반도체"],
    r"현대": [r"자동차|\b차\b"],
    r"두산": [r"에너빌리티|중공업|원전|터빈"],
    r"라인": [r"네이버|라인야후|Yahoo"],
}

# ─────────────────────────────────────────────────────────────
# 유틸
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

def set_encoding(resp: requests.Response):
    enc = None
    ctype = resp.headers.get("Content-Type", "")
    m = re.search(r"charset=([\w\-]+)", ctype, re.I)
    if m:
        enc = m.group(1).lower()

    if not enc:
        head = resp.content[:8192]
        m = re.search(br'charset=["\']?([\w\-]+)', head, re.I)
        if m:
            try:
                enc = m.group(1).decode("ascii", "ignore").lower()
            except Exception:
                enc = None

    if not enc:
        enc = (getattr(resp, "apparent_encoding", None) or "utf-8").lower()

    # 네이버 금융은 euc-kr/ms949/x-windows-949 계열이 흔함 → cp949로 통일
    if enc in ("euc-kr", "ks_c_5601-1987", "x-windows-949", "ms949", "cp949"):
        resp.encoding = "cp949"
    else:
        resp.encoding = enc


def clean_title(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"\([^)]+\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_wdate(s: str) -> Optional[datetime]:
    # 예: 2025-09-06 21:51:14
    s = (s or "").strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    except Exception:
        return None

# ── 거래일 캘린더 (휴장일/주말 보정) ───────────────────────────
def load_trading_days_from_prices(prices_csv: str) -> list[datetime.date]:
    px = pd.read_csv(prices_csv, parse_dates=["date"])
    days = sorted(px["date"].dt.date.unique())
    return days

def next_trading_day(d: datetime.date, trading_days: list[datetime.date]) -> Optional[datetime.date]:
    i = bisect.bisect_left(trading_days, d)
    return trading_days[i] if i < len(trading_days) else None

def market_ceiling_to_trading_day(dt: datetime, trading_days: list[datetime.date]) -> Optional[datetime.date]:
    base = dt.astimezone(KST).date()
    if dt.astimezone(KST).time() > MARKET_CLOSE:
        base = base + timedelta(days=1)
    return next_trading_day(base, trading_days)

# ─────────────────────────────────────────────────────────────
# 크롤링(카드형 레이아웃)
def fetch_mainnews_between(start_date: datetime.date,
                           end_date: datetime.date) -> pd.DataFrame:
    """
    날짜별 페이지 (?date=YYYY-MM-DD&page=N) 순회,
    .mainNewsList .newsList > li 구조에서 제목/언론사/일시 수집
    (여기서는 trade_date를 붙이지 않음; main에서 거래일로 매핑)
    """
    rows = []
    sess = requests.Session()

    cur = end_date
    while cur >= start_date:
        ymd = cur.strftime("%Y-%m-%d")
        day_total = 0

        for page in range(1, MAX_PAGES_PER_DAY + 1):
            try:
                r = sess.get(
                    MAINNEWS_URL,
                    params={"date": ymd, "page": page},
                    headers=HEADERS, timeout=10, allow_redirects=True
                )
            except Exception as e:
                print(f"[REQ FAIL] {ymd} p{page}: {e}", flush=True)
                break

            set_encoding(r)
            html_text = r.text
            if r.status_code != 200 or not html_text:
                break

            soup = BeautifulSoup(html_text, "html.parser")
            lis = soup.select("div.mainNewsList ul.newsList > li")
            if not lis:
                if page == 1 and day_total == 0:
                    with open(os.path.join(DEBUG_DIR, f"mainnews_{ymd}_p1.html"), "w", encoding="utf-8") as f:
                        f.write(html_text)
                    print(f"[DEBUG] saved {DEBUG_DIR}/mainnews_{ymd}_p1.html", flush=True)
                break

            page_rows = []
            for li in lis:
                a = li.select_one(".articleSubject a")
                if not a or not a.get("href"):
                    continue
                title = a.get_text(strip=True)
                url = urljoin(MAINNEWS_URL, a["href"])

                press = ""
                wdate_txt = ""
                summ = li.select_one(".articleSummary")
                if summ:
                    p = summ.select_one(".press")
                    press = p.get_text(strip=True) if p else ""
                    w = summ.select_one(".wdate")
                    wdate_txt = w.get_text(strip=True) if w else ""

                dt = parse_wdate(wdate_txt)
                if not title or not dt:
                    continue

                page_rows.append({"dt": dt, "title": title, "url": url, "press": press})

            if not page_rows:
                break

            rows.extend(page_rows)
            day_total += len(page_rows)
            time.sleep(REQUEST_SLEEP)

        print(f"[DAY] {ymd}: {day_total} rows", flush=True)
        cur -= timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["dt", "title", "url", "press"])

    df = (pd.DataFrame(rows)
            .drop_duplicates(subset=["title", "dt"])
            .sort_values("dt")
            .reset_index(drop=True))
    df["title_clean"] = df["title"].apply(clean_title)
    return df

# ─────────────────────────────────────────────────────────────
# 키워드 매칭 → 종목별 분리/저장
def match_company(title: str, patterns: List[str]) -> bool:
    t = title
    for broad, needers in CO_OCCUR.items():
        if re.search(broad, t, re.I) and not any(re.search(n, t, re.I) for n in needers):
            t = re.sub(broad, "", t, flags=re.I)
    return any(re.search(p, t, re.I) for p in patterns)

def split_by_company(df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        pats = KEYWORDS.get(tkr, [])
        if not pats:
            out[tkr] = pd.DataFrame(columns=df.columns)
            continue
        mask = df["title_clean"].apply(lambda s: match_company(s, pats))
        sub = df[mask].copy()
        if not sub.empty:
            sub["ticker"] = tkr
            sub["code"] = TICKER_TO_CODE.get(tkr, "")
        out[tkr] = sub
        print(f"[MATCH] {tkr}: {len(sub)} rows", flush=True)
    return out

def save_results(per_ticker: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ensure_dirs()
    merged = []
    for tkr, g in per_ticker.items():
        if g is None or g.empty:
            continue
        code = TICKER_TO_CODE.get(tkr, tkr)
        out_path = os.path.join(OUT_DIR, f"{code}.csv")
        g.to_csv(out_path, index=False, encoding="utf-8-sig")
        merged.append(g)
        print(f"  -> saved {len(g)} rows to {out_path}", flush=True)

    if not merged:
        print("[WARN] No matched news to save.", flush=True)
        return pd.DataFrame()

    all_df = pd.concat(merged, ignore_index=True)
    all_df.to_csv(os.path.join(OUT_DIR, "news_major_all.csv"), index=False, encoding="utf-8-sig")
    (all_df.groupby(["ticker", "trade_date"])["title"]
          .count().rename("news_cnt").reset_index()
          .to_csv(os.path.join(OUT_DIR, "news_cnt_daily.csv"), index=False, encoding="utf-8-sig"))
    print("[MERGED] -> data/news_major/news_major_all.csv", flush=True)
    print("[AGG]    -> data/news_major/news_cnt_daily.csv", flush=True)
    return all_df

# ─────────────────────────────────────────────────────────────
def main():
    global REQUEST_SLEEP

    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=str, default="data/raw/prices.csv",
                    help="기간/거래일 캘린더용 CSV (date,ticker,open,high,low,close,volume)")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (옵션)")
    ap.add_argument("--end",   type=str, default=None, help="YYYY-MM-DD (옵션)")
    ap.add_argument("--sleep", type=float, default=REQUEST_SLEEP, help="요청 간 딜레이(초)")
    ap.add_argument("--stream", action="store_true", help="일자별 즉시 매핑/저장(옵션)")
    args = ap.parse_args()

    REQUEST_SLEEP = float(args.sleep)

    # 기간 결정
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date   = datetime.strptime(args.end,   "%Y-%m-%d").date()
    else:
        prices_df = pd.read_csv(args.prices, parse_dates=["date"])
        start_date = prices_df["date"].min().date()
        end_date   = prices_df["date"].max().date()

    print(f"[RANGE] {start_date} ~ {end_date}", flush=True)

    # 거래일 캘린더(휴장일/주말 보정용)
    trading_days = load_trading_days_from_prices(args.prices)

    if args.stream:
        # 스트리밍 모드: 하루씩 수집 → 즉시 매핑/저장
        cur = end_date
        while cur >= start_date:
            ymd = cur.strftime("%Y-%m-%d")
            day_df = fetch_mainnews_between(cur, cur)
            if not day_df.empty:
                # 거래일 기준 trade_date 부여
                day_df["trade_date"] = day_df["dt"].apply(lambda x: market_ceiling_to_trading_day(x, trading_days))
                day_df = day_df.dropna(subset=["trade_date"])
                day_df["trade_date"] = pd.to_datetime(day_df["trade_date"])

                per_ticker = split_by_company(day_df, TICKERS)
                # append 저장 (헤더 자동 처리)
                ensure_dirs()
                for tkr, g in per_ticker.items():
                    if g is None or g.empty:
                        continue
                    code = TICKER_TO_CODE.get(tkr, tkr)
                    out = os.path.join(OUT_DIR, f"{code}.csv")
                    write_header = not os.path.exists(out)
                    g.to_csv(out, mode="a", header=write_header, index=False, encoding="utf-8-sig")
                print(f"[STREAM] saved {sum(len(v) for v in per_ticker.values())} rows on {ymd}", flush=True)
            cur -= timedelta(days=1)

        # 최종 병합/집계 파일 만들기
        files = [os.path.join(OUT_DIR, f"{TICKER_TO_CODE[t]}.csv") for t in TICKERS if os.path.exists(os.path.join(OUT_DIR, f"{TICKER_TO_CODE[t]}.csv"))]
        if files:
            all_df = pd.concat([pd.read_csv(p, parse_dates=["dt","trade_date"]) for p in files], ignore_index=True)
            all_df.to_csv(os.path.join(OUT_DIR, "news_major_all.csv"), index=False)
            (all_df.groupby(["ticker","trade_date"])["title"].count().rename("news_cnt").reset_index()
                 .to_csv(os.path.join(OUT_DIR, "news_cnt_daily.csv"), index=False))
            print("[MERGED] -> data/news_major/news_major_all.csv", flush=True)
            print("[AGG]    -> data/news_major/news_cnt_daily.csv", flush=True)
        return

    # 배치 모드: 전 기간 수집 → 한번에 매핑/저장
    news = fetch_mainnews_between(start_date, end_date)
    if news.empty:
        print("No mainnews parsed. Check selectors/date range.", flush=True)
        return

    # 거래일 기준 trade_date 부여(휴장일/주말/마감후 보정)
    news["trade_date"] = news["dt"].apply(lambda x: market_ceiling_to_trading_day(x, trading_days))
    news = news.dropna(subset=["trade_date"])
    news["trade_date"] = pd.to_datetime(news["trade_date"])

    per_ticker = split_by_company(news, TICKERS)
    save_results(per_ticker)

if __name__ == "__main__":
    main()
