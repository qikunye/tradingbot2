#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Trading Bot 

What it does
------------
- Per-ticker XGBoost classifier (predicts next-day up/down)
- Per-ticker LSTM regressor (predicts next-day close)
- Simple ensemble of the two
- Equal-weight portfolio across tickers
- Signals are lagged one bar to avoid look-ahead
- Saves a CSV of results and an equity curve plot

Quickstart
----------
python tradingbot.py \
  --tickers "SPY,AAPL,MSFT" \
  --start 2015-01-01 --end 2023-12-31 \
  --alpha 0.5 --tx_cost_bps 10

"""

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Silence verbose TF logs (comment out if you want all logs)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf        # noqa: E402
from tensorflow import keras   # noqa: E402
from tensorflow.keras import layers  # noqa: E402


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ml-bot")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig

def rolling_vol(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window, min_periods=window).std()


# -----------------------------------------------------------------------------
# Backtest stats
# -----------------------------------------------------------------------------
@dataclass
class BacktestStats:
    accuracy: float
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float


def calc_cagr(equity: pd.Series, periods_per_year=252) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def calc_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year=252) -> float:
    mu, sigma = returns.mean(), returns.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return ((mu - rf) / sigma) * math.sqrt(periods_per_year)

def calc_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


# -----------------------------------------------------------------------------
# Data & features
# -----------------------------------------------------------------------------
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV. Ensure we have a single 'Close'."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns=str.title)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic TA features + next-day targets."""
    out = df.copy()

    out["SMA10"] = sma(out["Close"], 10)
    out["SMA50"] = sma(out["Close"], 50)
    out["RSI14"] = rsi(out["Close"], 14)

    macd_line, macd_sig = macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd_line
    out["MACD_signal"] = macd_sig
    out["Volatility20"] = rolling_vol(out["Close"], 20)

    # Targets: up/down and next close
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out["NextClose"] = out["Close"].shift(-1)

    out = out.dropna().copy()
    return out

def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# -----------------------------------------------------------------------------
# XGBoost (classification)
# -----------------------------------------------------------------------------
def train_xgb(train: pd.DataFrame, features: list) -> "xgb.XGBClassifier":
    """Train a simple but decent XGB classifier with mild class-imbalance handling."""
    import inspect

    X = train[features].values
    y = train["Target"].values

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    spw = float(neg / max(pos, 1))  # avoid division by 0

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        eval_metric="logloss",
    )

    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs = {"X": X_tr, "y": y_tr, "eval_set": [(X_val, y_val)], "verbose": False}

    # Early stopping if this XGBoost version supports it
    if "early_stopping_rounds" in fit_sig:
        fit_kwargs["early_stopping_rounds"] = 80

    model.fit(**fit_kwargs)
    return model

def xgb_labels(model: "xgb.XGBClassifier", X_df: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_df.values)[:, 1]
    return (proba >= 0.5).astype(int).astype(float)


# -----------------------------------------------------------------------------
# LSTM (regression)
# -----------------------------------------------------------------------------
def make_lstm_dataset(df: pd.DataFrame, features: List[str], target_col: str, seq_len: int = 60):
    """Return (X_seq, y_seq, scaler). X_seq shape: (N, seq_len, n_features)."""
    scaler = StandardScaler()
    feat = df[features].values.astype("float32")
    feat_scaled = scaler.fit_transform(feat)

    y = df[target_col].values.astype("float32")
    X_seq, y_seq = [], []
    for i in range(len(df) - seq_len):
        X_seq.append(feat_scaled[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    X_seq = np.array(X_seq, dtype="float32")
    y_seq = np.array(y_seq, dtype="float32")
    return X_seq, y_seq, scaler

def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

def train_lstm(train_df: pd.DataFrame, features: List[str],
               seq_len: int = 60, epochs: int = 30, batch_size: int = 32):
    """Train a small LSTM to predict next-day close."""
    X, y, scaler = make_lstm_dataset(train_df, features, "NextClose", seq_len=seq_len)
    if len(X) < 20:
        return None, None

    val_cut = int(len(X) * 0.85)
    X_tr, y_tr = X[:val_cut], y[:val_cut]
    X_val, y_val = X[val_cut:], y[val_cut:]

    model = build_lstm((X.shape[1], X.shape[2]))
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
    return model, scaler

def lstm_predict_next_close(model, scaler: StandardScaler, df: pd.DataFrame,
                            features: List[str], seq_len: int = 60):
    """Walk-forward prediction of next close from the last seq_len window."""
    if (model is None) or (scaler is None) or (len(df) < seq_len + 1):
        return None
    feat = df[features].values.astype("float32")
    feat_scaled = scaler.transform(feat)
    X = np.expand_dims(feat_scaled[-seq_len:], axis=0)
    pred = model.predict(X, verbose=0)[0, 0]
    return float(pred)


# -----------------------------------------------------------------------------
# Per-ticker model bundle
# -----------------------------------------------------------------------------
@dataclass
class TickerModels:
    xgb: xgb.XGBClassifier | None
    lstm: keras.Model | None
    scaler: StandardScaler | None


def train_models_for_ticker(ticker: str, feat_df: pd.DataFrame, features: List[str],
                            seq_len: int, lstm_epochs: int, lstm_batch: int,
                            use_xgb: bool, use_lstm: bool) -> TickerModels:
    """Train both models on the first 80% of data; report XGB accuracy on the last 20%."""
    train_df, test_df = time_split(feat_df, 0.8)

    model_xgb = None
    if use_xgb:
        model_xgb = train_xgb(train_df, features)
        y_pred = xgb_labels(model_xgb, test_df[features])
        acc = accuracy_score(test_df["Target"], y_pred)
        log.info("[XGB] %-5s accuracy (OOS): %.3f", ticker, acc)

    model_lstm = scaler = None
    if use_lstm:
        model_lstm, scaler = train_lstm(train_df, features,
                                        seq_len=seq_len, epochs=lstm_epochs, batch_size=lstm_batch)
        if model_lstm is None:
            log.warning("[LSTM] %s skipped (not enough data).", ticker)

    return TickerModels(model_xgb, model_lstm, scaler)


# -----------------------------------------------------------------------------
# Column helpers for robustness
# -----------------------------------------------------------------------------
def pick_close_column(df: pd.DataFrame, ticker: str) -> str:
    """
    Find a usable 'close' column even if columns were flattened or suffixed.
    Preference order:
      1) 'Close' (case-insensitive)
      2) endswith '_Close'
      3) contains 'close'
    """
    cols = list(df.columns)
    lows = [str(c).lower() for c in cols]

    if "close" in lows:
        return cols[lows.index("close")]

    for c in cols:
        if str(c).lower().endswith("_close"):
            return c

    for c in cols:
        if "close" in str(c).lower():
            return c

    raise RuntimeError(f"No close-like column found for {ticker}. Columns: {list(df.columns)}")


# -----------------------------------------------------------------------------
# Portfolio backtest
# -----------------------------------------------------------------------------
def backtest_portfolio(all_feat: Dict[str, pd.DataFrame],
                       all_models: Dict[str, TickerModels],
                       features: List[str],
                       seq_len: int = 60,
                       tx_cost_bps: float = 10.0,       # 0.10% per position change
                       ensemble_alpha: float = 0.5,
                       use_xgb: bool = True,
                       use_lstm: bool = True) -> Tuple[pd.DataFrame, BacktestStats]:
    """
    Build per-ticker signals, then combine into an equal-weight portfolio.
    Signals are generated out-of-sample (last 20%) and lagged by one bar.
    """

    def _flatten_cols(cols) -> List[str]:
        if isinstance(cols, pd.MultiIndex):
            return [
                "_".join(str(x) for x in tup if x is not None and str(x).strip() != "")
                for tup in cols
            ]
        return [str(c) for c in cols]

    def _norm_name(s: str) -> str:
        s = str(s).strip()
        s = s.replace("(", "").replace(")", "").replace("'", "").replace("\"", "").replace(" ", "")
        if "," in s:
            s = s.split(",")[0]
        return s.lower()

    def map_features(cols, wanted):
        """Map 'SMA10' etc. to actual column labels after any flattening/suffixing."""
        norm_to_actual = {}
        for c in cols:
            n = _norm_name(c)
            if n not in norm_to_actual:
                norm_to_actual[n] = c
        actual_cols = []
        for feat in wanted:
            key = feat.lower()
            if key in norm_to_actual:
                actual_cols.append(norm_to_actual[key])
                continue
            # allow prefixes like "sma10_spy"
            hit = next((c for n, c in norm_to_actual.items() if n.startswith(key + "_")), None)
            if hit is None:
                raise KeyError(f"Feature '{feat}' not found. Have: {list(cols)}")
            actual_cols.append(hit)
        return actual_cols

    per_ticker: Dict[str, pd.DataFrame] = {}
    oos_start_by_ticker: Dict[str, pd.Timestamp] = {}

    # ----- per-ticker signals -----
    for ticker, df in all_feat.items():
        m = all_models[ticker]
        out = df.copy().sort_index()
        out.index = pd.to_datetime(out.index)
        if isinstance(out.index, pd.MultiIndex):
            out.index = out.index.get_level_values(0)

        # Flatten columns and remove accidental duplicates
        out.columns = _flatten_cols(out.columns)
        if pd.Index(out.columns).duplicated().any():
            out = out.loc[:, ~pd.Index(out.columns).duplicated()]

        # Returns off a robust 'close'
        close_col = pick_close_column(out, ticker)
        out["Daily_Return"] = out[close_col].pct_change().fillna(0.0)

        # Map features safely (post-flatten)
        feature_cols = map_features(out.columns, features)

        # Unified OOS cut (80/20)
        cut_idx = int(len(out) * 0.8)
        if cut_idx >= len(out):
            log.warning("[DATA] %s not enough rows for OOS split.", ticker)
            continue
        oos_start_by_ticker[ticker] = out.index[cut_idx]

        # XGB signals (OOS only)
        xgb_sig = np.zeros(len(out), dtype=float)
        has_xgb = bool(use_xgb and (m.xgb is not None))
        if has_xgb:
            X_oos = out[feature_cols].iloc[cut_idx:]
            sig_oos = xgb_labels(m.xgb, X_oos)
            xgb_sig[cut_idx:cut_idx + len(sig_oos)] = sig_oos

        # LSTM signals (walk-forward, OOS only)
        lstm_sig = np.zeros(len(out), dtype=float)
        has_lstm = bool(use_lstm and (m.lstm is not None) and (m.scaler is not None))
        if has_lstm:
            vals = out[close_col].to_numpy()
            for i in range(cut_idx, len(out) - 1):
                hist = out.iloc[: i + 1]
                pred = lstm_predict_next_close(m.lstm, m.scaler, hist, feature_cols, seq_len)
                if pred is None:
                    lstm_sig[i] = 0.0
                else:
                    today_close = float(vals[i] if vals.ndim == 1 else vals[i, 0])
                    lstm_sig[i] = 1.0 if float(pred) > today_close else 0.0
            lstm_sig[-1] = 0.0  # cannot trade the last bar

        # Blend
        if has_xgb and has_lstm:
            blend = ensemble_alpha * xgb_sig + (1 - ensemble_alpha) * lstm_sig
            signal = (blend >= 0.5).astype(float)
            long_share = float(np.mean(signal[cut_idx:])) if cut_idx < len(signal) else 0.0
            log.info("[SIG] %-5s long-share=%.3f (xgb=%s, lstm=%s)", ticker, long_share, has_xgb, has_lstm)
        elif has_xgb:
            signal = xgb_sig
        elif has_lstm:
            signal = lstm_sig
        else:
            signal = np.zeros(len(out), dtype=float)

        # Today’s position = yesterday’s signal
        out["Position"] = pd.Series(signal, index=out.index).shift(1).fillna(0.0)

        # Simple transaction costs (per change in position)
        if tx_cost_bps > 0:
            pos_chg = out["Position"].diff().abs().fillna(out["Position"].abs())
            out["Tx_Cost"] = -(tx_cost_bps / 10000.0) * pos_chg
        else:
            out["Tx_Cost"] = 0.0

        out["Strategy_Return"] = out["Position"] * out["Daily_Return"] + out["Tx_Cost"]
        out["Equity"] = (1 + out["Strategy_Return"]).cumprod()

        per_ticker[ticker] = out[["Daily_Return", "Strategy_Return", "Equity", "Target"]]

    # If nothing made it through, bail gracefully
    if not oos_start_by_ticker:
        log.warning("No tickers produced signals. Returning empty results.")
        empty = pd.DataFrame()
        stats = BacktestStats(accuracy=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0, total_return=0.0)
        return empty, stats

    # Align to a common OOS window across all tickers
    common_start = max(oos_start_by_ticker.values())
    frames = []
    for t, df in per_ticker.items():
        trimmed = df.loc[df.index >= common_start].copy()
        renamed = trimmed.rename(columns={
            "Daily_Return": f"{t}_ret",
            "Strategy_Return": f"{t}_strat",
            "Equity": f"{t}_eq",
            "Target": f"{t}_tgt",
        })
        renamed.index = pd.to_datetime(renamed.index)
        if isinstance(renamed.index, pd.MultiIndex):
            renamed.index = renamed.index.get_level_values(0)
        frames.append(renamed)

    if not frames:
        log.warning("No frames to join after OOS alignment.")
        empty = pd.DataFrame()
        stats = BacktestStats(accuracy=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0, total_return=0.0)
        return empty, stats

    panel = pd.concat(frames, axis=1, join="inner").sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    strat_cols = [c for c in panel.columns if c.endswith("_strat")]
    bh_cols = [c for c in panel.columns if c.endswith("_ret")]

    if not strat_cols:
        log.warning("No *_strat columns found — check joins/column names.")
        panel["Portfolio_Strategy_Return"] = 0.0
        panel["Portfolio_BuyHold_Return"] = 0.0
    else:
        panel["Portfolio_Strategy_Return"] = panel[strat_cols].mean(axis=1)
        panel["Portfolio_BuyHold_Return"] = panel[bh_cols].mean(axis=1) if bh_cols else 0.0

    panel["Portfolio_Equity"] = (1 + panel["Portfolio_Strategy_Return"]).cumprod()
    panel["BuyHold_Equity"] = (1 + panel["Portfolio_BuyHold_Return"]).cumprod()

    if panel["Portfolio_Strategy_Return"].abs().sum() == 0:
        log.warning("All portfolio strategy returns are zero — check signals/joins.")

    # Rough “hit rate” proxy per ticker
    accs = []
    for t in all_feat.keys():
        tgt_col = f"{t}_tgt"
        strat_col = f"{t}_strat"
        if tgt_col in panel.columns and strat_col in panel.columns:
            tgt = panel[tgt_col].astype(int)
            label = (panel[strat_col].shift(-1).fillna(0.0) > 0).astype(int)
            if len(tgt) and len(label):
                accs.append(accuracy_score(tgt, label))
    accuracy = float(np.mean(accs)) if accs else 0.0

    eq = panel["Portfolio_Equity"].dropna()
    rets = panel["Portfolio_Strategy_Return"].dropna()

    stats = BacktestStats(
        accuracy=accuracy,
        cagr=calc_cagr(eq),
        sharpe=calc_sharpe(rets),
        max_drawdown=calc_max_drawdown(eq),
        total_return=float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
    )
    return panel, stats


# -----------------------------------------------------------------------------
# Plotting & CLI
# -----------------------------------------------------------------------------
def plot_equity(panel: pd.DataFrame, out_path: str = "equity_curve.png") -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(panel.index, panel["Portfolio_Equity"], label="Strategy (Equal-Weight)")
    plt.plot(panel.index, panel["BuyHold_Equity"], label="Buy & Hold (Equal-Weight)")
    plt.title("ML Strategy vs Equal-Weight Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    log.info("Saved: %s", out_path)


def main():
    p = argparse.ArgumentParser(description="XGBoost + LSTM (multi-asset) ML bot")
    p.add_argument("--tickers", type=str, default="SPY,AAPL,MSFT",
                   help="Comma-separated list, e.g. 'SPY,AAPL,MSFT'")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default="2023-12-31")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--tx_cost_bps", type=float, default=0.0)
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.5, help="Ensemble weight on XGB (0..1)")
    p.add_argument("--xgb_only", action="store_true")
    p.add_argument("--lstm_only", action="store_true")
    p.add_argument("--out_csv", type=str, default="backtest_results.csv")
    p.add_argument("--out_png", type=str, default="equity_curve.png")
    args = p.parse_args()

    use_xgb = not args.lstm_only
    use_lstm = not args.xgb_only
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    log.info("Downloading %s  %s → %s", tickers, args.start, args.end)
    features = ["SMA10", "SMA50", "RSI14", "MACD", "MACD_signal", "Volatility20"]

    # Build features per ticker
    feat_by_ticker: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        raw = download_data(t, args.start, args.end)
        feat_by_ticker[t] = make_features(raw)

    # Train models per ticker
    models_by_ticker: Dict[str, TickerModels] = {}
    for t in tickers:
        models_by_ticker[t] = train_models_for_ticker(
            t, feat_by_ticker[t], features,
            seq_len=args.seq_len, lstm_epochs=args.epochs, lstm_batch=args.batch_size,
            use_xgb=use_xgb, use_lstm=use_lstm
        )

    # Backtest equal-weight portfolio
    panel, stats = backtest_portfolio(
        feat_by_ticker, models_by_ticker, features,
        seq_len=args.seq_len, tx_cost_bps=args.tx_cost_bps,
        ensemble_alpha=args.alpha, use_xgb=use_xgb, use_lstm=use_lstm
    )

    panel.to_csv(args.out_csv)
    log.info("Saved: %s", args.out_csv)
    plot_equity(panel, args.out_png)

    log.info(
        "\n=== Portfolio Backtest ===\n"
        "Tickers:       %s\n"
        "Total Return:  %6.2f%%\n"
        "CAGR:          %6.2f%%\n"
        "Sharpe:        %6.2f\n"
        "Max Drawdown:  %6.2f%%\n"
        "Hit-Rate est.: %6.2f%%",
        ", ".join(tickers),
        stats.total_return * 100,
        stats.cagr * 100,
        stats.sharpe,
        stats.max_drawdown * 100,
        stats.accuracy * 100,
    )


if __name__ == "__main__":
    # Some reproducibility (don’t overthink this; just keeps runs stable enough)
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    main()
