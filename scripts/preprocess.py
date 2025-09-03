"""
전처리 파이프라인:
- 입력: data/raw/prices.csv (date,ticker,open,high,low,close,volume)
- 단계: 정렬/캘린더정렬 -> 라벨(y=log close_{t+H}/close_t) -> 피처 생성
        -> 누수방지 스케일링(Train 통계만) -> 시간 분할 -> 시퀀스 윈도우 저장
- 출력:
  1) 패널 parquet: panel_{split}.parquet  (unique_id, ds, y, feat_*)
  2) 윈도우 npz:   windows_{split}.npz    (X[L,F], y, ids, ds)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)

def read_prices(path: str) -> pd.DataFrame:
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["date"])
    need = {"date","ticker","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        raise ValueError(f"prices.csv must have columns: {sorted(list(need))}")

    # 숫자 컬럼을 강제 float 변환(쉼표/공백/문자 포함 케이스 방어)
    num_cols = ["open","high","low","close","volume"]
    for c in num_cols:
        # 먼저 문자열로 변환 후 공백 제거
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 필수값 결측 제거
    df = df.dropna(subset=["close"])

    # 정렬
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    return df


def align_common_calendar(df: pd.DataFrame) -> pd.DataFrame:
    # 티커별 날짜 교집합(모든 종목이 존재하는 공통 영업일만 채택)
    dates_sets = df.groupby("ticker")["date"].apply(set).tolist()
    common = sorted(list(set.intersection(*dates_sets)))
    out = df[df["date"].isin(common)].copy()
    # 안전 차원으로 재정렬
    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)

def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = df.groupby("ticker", group_keys=False)

    # 라벨: H일 후 로그수익률
    df["close_fwd"] = g["close"].shift(-horizon)
    df["y"] = np.log(df["close_fwd"] / df["close"])

    # 1일 로그수익률
    df["ret1"] = g["close"].apply(lambda s: np.log(s).diff())

    # 변동성: 20일 표준편차(비모수)
    df["sigma20"] = g["ret1"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)

    # 이동평균/모멘텀류(필요 최소)
    df["ma10"]  = g["close"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["ma20"]  = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["mom10"] = g["close"].transform(lambda s: s / s.shift(10) - 1.0)

    # 거래대금 z-score
    df["amt"]   = df["close"] * df["volume"]
    mu_amt      = g["amt"].transform("mean")
    std_amt     = g["amt"].transform("std").replace(0, np.nan)
    df["amt_z"] = (df["amt"] - mu_amt) / (std_amt + 1e-12)

    # RSI
    df["rsi14"] = g["close"].transform(lambda s: rsi(np.log(s+1e-12), 14))

    # 결측 처리(초기 구간/휴장 영향): 앞뒤 보간
    for col in ["ret1","sigma20","ma10","ma20","mom10","amt_z","rsi14"]:
        df[col] = g[col].apply(lambda s: s.bfill().ffill())

    return df

# ──────────────────────────────────────────────────────────────────────────────
# Split + scaling (leakage-safe)
def split_df(df: pd.DataFrame, train_end: str, val_end: str):
    d_train_end = pd.to_datetime(train_end)
    d_val_end   = pd.to_datetime(val_end)
    tr = df[df["date"] <= d_train_end].copy()
    va = df[(df["date"] > d_train_end) & (df["date"] <= d_val_end)].copy()
    te = df[df["date"] > d_val_end].copy()
    return tr, va, te

def fit_scaler(train: pd.DataFrame, feat_cols: list, by_ticker=True):
    stats = {}
    if by_ticker:
        for tkr, g in train.groupby("ticker"):
            mu = g[feat_cols].mean().to_dict()
            sd = g[feat_cols].std().replace(0, np.nan).to_dict()
            stats[tkr] = {"mean": mu, "std": {k: float(sd[k]) if pd.notnull(sd[k]) else 1.0 for k in sd}}
    else:
        mu = train[feat_cols].mean().to_dict()
        sd = train[feat_cols].std().replace(0, np.nan).to_dict()
        stats["__global__"] = {"mean": mu, "std": {k: float(sd[k]) if pd.notnull(sd[k]) else 1.0 for k in sd}}
    return stats

def apply_scaler(df: pd.DataFrame, stats: dict, feat_cols: list, by_ticker=True):
    out = df.copy()
    if by_ticker:
        for tkr, g in out.groupby("ticker"):
            par = stats.get(tkr, None)
            if par is None:
                # 티커가 train에 없었던 경우(일반적으론 없음) → 글로벌 평균 사용
                par = stats.get("__global__", {"mean":{c:0. for c in feat_cols},
                                               "std": {c:1. for c in feat_cols}})
            mu = par["mean"]
            sd = par["std"]
            for c in feat_cols:
                out.loc[g.index, c] = (g[c] - mu[c]) / (sd[c] + 1e-12)
    else:
        par = stats["__global__"]
        mu = par["mean"]; sd = par["std"]
        out[feat_cols] = (out[feat_cols] - pd.Series(mu)) / (pd.Series(sd) + 1e-12)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Panel + Windows
def save_panel(df: pd.DataFrame, path: str, feat_cols: list):
    panel = df.rename(columns={"ticker":"unique_id","date":"ds"})[["unique_id","ds","y", *feat_cols]]
    panel.to_parquet(path, index=False)

def make_windows(df: pd.DataFrame, seq_len: int, feat_cols: list):
    """
    각 티커별로 슬라이딩 윈도우 생성
    X: [L, F], y: 스텝 t의 타겟(=미리 계산된 y)
    """
    Xs, ys, ids, dss = [], [], [], []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("date")
        vals = g[feat_cols].values
        target = g["y"].values
        dates = g["date"].values
        for i in range(len(g) - seq_len):
            x = vals[i:i+seq_len]
            y = target[i+seq_len-1]  # 윈도우 마지막 시점의 타겟
            if np.isfinite(y):
                Xs.append(x.astype(np.float32))
                ys.append(np.float32(y))
                ids.append(tkr)
                dss.append(dates[i+seq_len-1])
    Xs = np.stack(Xs) if Xs else np.zeros((0, seq_len, len(feat_cols)), dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    ids = np.array(ids)
    dss = np.array(dss, dtype="datetime64[ns]")
    return Xs, ys, ids, dss

def save_windows(df: pd.DataFrame, seq_len: int, feat_cols: list, out_path: str):
    X, y, ids, dss = make_windows(df, seq_len, feat_cols)
    np.savez_compressed(out_path, X=X, y=y, ids=ids, ds=dss)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True, help="data/raw/prices.csv")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--train-end", type=str, required=True)
    ap.add_argument("--val-end", type=str, required=True)
    ap.add_argument("--by-ticker-scaling", action="store_true", help="티커별 표준화(권장)")
    args = ap.parse_args()

    ensure_dirs()
    df = read_prices(args.prices)
    df = align_common_calendar(df)
    df = add_features(df, args.horizon)

    # 학습에 필요한 컬럼만 유지
    keep_cols = ["ticker","date","y","sigma20","ma10","ma20","mom10","amt_z","rsi14"]
    df = df[keep_cols].dropna(subset=["y"]).copy()

    # 시간 분할
    tr, va, te = split_df(df, args.train_end, args.val_end)

    # 스케일링(Train 통계만)
    feat_cols = ["sigma20","ma10","ma20","mom10","amt_z","rsi14"]
    stats = fit_scaler(tr, feat_cols, by_ticker=args.by_ticker_scaling)
    tr_s = apply_scaler(tr, stats, feat_cols, by_ticker=args.by_ticker_scaling)
    va_s = apply_scaler(va, stats, feat_cols, by_ticker=args.by_ticker_scaling)
    te_s = apply_scaler(te, stats, feat_cols, by_ticker=args.by_ticker_scaling)

    # 패널 저장 (NeuralForecast 호환)
    save_panel(tr_s, "data/processed/panel_train.parquet", feat_cols)
    save_panel(va_s, "data/processed/panel_val.parquet",   feat_cols)
    save_panel(te_s, "data/processed/panel_test.parquet",  feat_cols)

    # 윈도우 저장 (PyTorch 학습용)
    save_windows(tr_s, args.seq_len, feat_cols, "data/processed/windows_train.npz")
    save_windows(va_s, args.seq_len, feat_cols, "data/processed/windows_val.npz")
    save_windows(te_s, args.seq_len, feat_cols, "data/processed/windows_test.npz")

    # 스케일러 파라미터 저장
    with open("data/processed/scaler_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved:")
    print(" - data/processed/panel_{train,val,test}.parquet")
    print(" - data/processed/windows_{train,val,test}.npz")
    print(" - data/processed/scaler_stats.json")

if __name__ == "__main__":
    main()
