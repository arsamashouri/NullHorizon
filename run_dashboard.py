#!/usr/bin/env python3
"""
NullHorizon v1.1: Precision Stochastic Market Forecaster
Powered by the Kronos Autoregressive Foundation Model for Candlestick Sequences.

Usage:
    python run_dashboard.py [--host 127.0.0.1] [--port 8088] [--symbol BTCUSDT] [--interval 1h]

v1.1 Updates:
    - Left-hand layout: Configuration menu positioned on the left side of the screen.
    - Single-screen fit: Tight margins and adaptive viewport calculations (100vh zero vertical scroll).
    - Refined, non-neon color system: Subdued dark slate / obsidian with muted forest sage and terracotta.
    - Light / Dark mode toggle with real-time TradingView chart canvas theme switching.
    - Version bumped to v1.1.
    - Zero system log clutter.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch
from flask import Flask, jsonify, render_template_string, request
from model import Kronos, KronosTokenizer, KronosPredictor

# ----------------------------- Configuration -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "weights", "Kronos-base")
LOCAL_TOKENIZER_DIR = os.path.join(BASE_DIR, "weights", "Kronos-Tokenizer-base")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "4h"
DEFAULT_HORIZON = 48
DEFAULT_PATHS = 100
CONTEXT_LEN = 512
REFRESH_SECONDS = 900
MODEL_ID = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else "NeoQuasar/Kronos-base"
TOKENIZER_ID = LOCAL_TOKENIZER_DIR if os.path.isdir(LOCAL_TOKENIZER_DIR) else "NeoQuasar/Kronos-Tokenizer-base"
BINANCE_API = "https://data-api.binance.vision/api/v3/klines"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]

DELTA_MAP = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nullhorizon")

app = Flask(__name__)
lock = threading.RLock()
predictor: KronosPredictor | None = None
device_label = "CPU"

current_cycle_id = 0

state = {
    "status": "Initializing...",
    "error": None,
    "symbol": DEFAULT_SYMBOL,
    "interval": DEFAULT_INTERVAL,
    "model": "Kronos-base (Local)" if os.path.isdir(LOCAL_MODEL_DIR) else MODEL_ID,
    "device": "Detecting Accelerator",
    "context": CONTEXT_LEN,
    "horizon": DEFAULT_HORIZON,
    "target_paths": DEFAULT_PATHS,
    "completed_paths": 0,
    "progress": 0,
    "candles": [],
    "paths": [],
    "stats": {},
    "updated_at": None,
    "cycle": 0,
}


def timestamps_for(df: pd.DataFrame, interval: str, horizon: int):
    delta = DELTA_MAP.get(interval, pd.Timedelta(hours=1))
    last_dt = pd.to_datetime(df["time"].iloc[-1], utc=True)
    future_dts = [last_dt + delta * (step + 1) for step in range(horizon)]
    x_ts = pd.Series(df["time"])
    y_ts = pd.Series(future_dts)
    return x_ts, y_ts


def fetch_candles(symbol: str, interval: str) -> pd.DataFrame:
    params = {
        "symbol": symbol.strip().upper(),
        "interval": interval,
        "limit": min(1000, max(CONTEXT_LEN, 200)),
    }
    resp = requests.get(BINANCE_API, params=params, timeout=15)
    if resp.status_code != 200:
        err_msg = resp.text
        try:
            err_json = resp.json()
            err_msg = err_json.get("msg", err_msg)
        except Exception:
            pass
        raise RuntimeError(f"Binance API ({resp.status_code}): {err_msg}")

    rows = resp.json()
    if not rows or not isinstance(rows, list):
        raise RuntimeError(f"No candlestick data returned for symbol '{symbol}'.")

    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 7]].copy()
    df.columns = ["time"] + FEATURES
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def candle_payload(df: pd.DataFrame) -> list:
    out = []
    for _, row in df.tail(180).iterrows():
        out.append({
            "time": int(row["time"].timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return out


def publish_path(values: list[float], anchor_time: int, future_times: list[int]):
    points = [{"time": anchor_time, "value": values[0]}]
    for ts, val in zip(future_times, values):
        points.append({"time": ts, "value": val})

    with lock:
        state["paths"].append(points)
        count = len(state["paths"])
        state["completed_paths"] = count
        target = max(1, state["target_paths"])
        state["progress"] = min(100, int(round((count / target) * 100)))

        all_curves = np.array([
            [pt["value"] for pt in p[1:]] for p in state["paths"]
        ])
        last_close = state["stats"].get("last", values[0])
        terminal = all_curves[:, -1]

        p50 = float(np.median(terminal))
        p10 = float(np.percentile(terminal, 10))
        p90 = float(np.percentile(terminal, 90))
        up_count = int(np.sum(terminal > last_close))
        up_pct = float(round((up_count / count) * 100, 1))
        change_pct = float(round(((p50 - last_close) / (last_close + 1e-9)) * 100, 2))

        med_curve = np.median(all_curves, axis=0).tolist()
        p10_curve = np.percentile(all_curves, 10, axis=0).tolist()
        p90_curve = np.percentile(all_curves, 90, axis=0).tolist()

        state["stats"].update({
            "p50": p50,
            "p10": p10,
            "p90": p90,
            "up_pct": up_pct,
            "median_change_pct": change_pct,
            "partial_paths": count,
            "partial_median": med_curve,
            "partial_p10": p10_curve,
            "partial_p90": p90_curve,
        })


def run_forecast_cycle(force_cycle_id: int | None = None):
    global current_cycle_id
    if force_cycle_id is not None:
        my_cycle = force_cycle_id
    else:
        with lock:
            current_cycle_id += 1
            my_cycle = current_cycle_id

    with lock:
        sym = state["symbol"]
        itv = state["interval"]
        target_hz = state["horizon"]
        target_p = state["target_paths"]
        state.update({
            "status": f"Fetching {sym} ({itv})...",
            "error": None,
            "paths": [],
            "completed_paths": 0,
            "progress": 0,
            "cycle": state["cycle"] + 1,
        })

    logger.info(f"[Cycle {my_cycle}] Fetching market candles for {sym} ({itv})...")
    try:
        df = fetch_candles(sym, itv)
    except Exception as exc:
        logger.error(f"[Cycle {my_cycle}] Candle retrieval failed: {exc}")
        with lock:
            if current_cycle_id == my_cycle:
                state["status"] = "Fetch Error"
                state["error"] = str(exc)
        return

    if current_cycle_id != my_cycle:
        logger.info(f"[Cycle {my_cycle}] Superseded before inference.")
        return

    try:
        if len(df) < 32:
            raise RuntimeError(f"Insufficient history ({len(df)} candles returned).")

        df = df.tail(CONTEXT_LEN).reset_index(drop=True)
        x_df = df[FEATURES].copy()
        x_ts, y_ts = timestamps_for(df, itv, target_hz)

        last_close = float(df["close"].iloc[-1])
        anchor_time = int(df["time"].iloc[-1].timestamp())
        future_times = [int(t.timestamp()) for t in y_ts]

        with lock:
            if current_cycle_id != my_cycle:
                return
            state["candles"] = candle_payload(df)
            state["stats"] = {
                "last": last_close,
                "last_candle_time": df["time"].iloc[-1].isoformat(),
                "forecast_times": future_times,
                "p50": last_close,
                "p10": last_close,
                "p90": last_close,
                "up_pct": 0.0,
                "median_change_pct": 0.0,
                "partial_median": [last_close] * target_hz,
                "partial_p10": [last_close] * target_hz,
                "partial_p90": [last_close] * target_hz,
            }
            state["status"] = f"Simulating (0/{target_p})..."

        logger.info(f"[Cycle {my_cycle}] Data loaded. Last close: {last_close:,.4f}. Simulating {target_p} paths...")

        for i in range(target_p):
            if current_cycle_id != my_cycle:
                logger.info(f"[Cycle {my_cycle}] Aborted at path {i + 1}/{target_p}.")
                return

            with lock:
                state["status"] = f"Sampling path {i + 1}/{target_p}..."

            result = predictor.predict(
                df=x_df.copy(),
                x_timestamp=x_ts.copy(),
                y_timestamp=y_ts.copy(),
                pred_len=target_hz,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )

            if isinstance(result, pd.DataFrame):
                vals = result["close"].astype(float).tolist()
            elif isinstance(result, dict):
                vals = result["close"].astype(float).tolist()
            else:
                vals = result["close"].astype(float).tolist()

            if len(vals) != target_hz:
                raise RuntimeError(f"Expected {target_hz} forecast points, got {len(vals)}.")

            if current_cycle_id != my_cycle:
                return

            publish_path(vals, anchor_time, future_times)

        with lock:
            if current_cycle_id == my_cycle:
                s = state["stats"]
                s["regime"] = "BULLISH" if s["p50"] > last_close * 1.002 else (
                    "BEARISH" if s["p50"] < last_close * 0.998 else "CHOPPY"
                )
                s["up_count"] = int(round(s["up_pct"] * target_p / 100))
                s["path_count"] = target_p
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                state["status"] = "Live · Forecast Complete"

        logger.info(f"[Cycle {my_cycle}] Forecast finished for {sym} {itv} | Median: {state['stats']['p50']:,.4f}")

    except Exception as exc:
        logger.exception(f"[Cycle {my_cycle}] Forecast exception")
        with lock:
            if current_cycle_id == my_cycle:
                state["status"] = "Inference Error"
                state["error"] = str(exc)


def background_loop():
    while True:
        run_forecast_cycle()
        time.sleep(max(60, REFRESH_SECONDS))


@app.route("/api/state")
def get_state():
    with lock:
        return jsonify(state)


@app.route("/api/config", methods=["POST"])
def update_config():
    global current_cycle_id
    data = request.get_json(silent=True) or {}
    sym = data.get("symbol")
    itv = data.get("interval")
    hz = data.get("horizon")
    paths = data.get("paths")

    with lock:
        if sym and isinstance(sym, str):
            state["symbol"] = sym.strip().upper()
        if itv and itv in DELTA_MAP:
            state["interval"] = itv
        if hz and isinstance(hz, int) and 12 <= hz <= 500:
            state["horizon"] = hz
        if paths and isinstance(paths, int) and 10 <= paths <= 500:
            state["target_paths"] = paths

        current_cycle_id += 1
        new_cycle_id = current_cycle_id

    threading.Thread(target=run_forecast_cycle, args=(new_cycle_id,), daemon=True).start()
    return jsonify({"ok": True, "cycle": new_cycle_id, "config": {
        "symbol": state["symbol"],
        "interval": state["interval"],
        "horizon": state["horizon"],
        "paths": state["target_paths"],
    }})


# ----------------------------- Single-Screen UI (v1.1) -----------------------------
PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <title>NullHorizon v1.1 · Probabilistic Market Forecaster</title>
  <script src="/static/js/lightweight-charts.standalone.production.js"></script>
  <script>
    if (typeof LightweightCharts === 'undefined') {
      document.write('<script src="https://unpkg.com/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"><\/script>');
    }
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    /* ---------------- Design System: Non-Neon, Refined Palette ---------------- */
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #0c0f14;
      --panel: #131720;
      --panel-hover: #181d28;
      --card-inner: #0f1219;
      --line: #202634;
      --line-subtle: #191f2c;
      --text: #e2e7ef;
      --text-dim: #9aa5b5;
      --muted: #677485;
      --brand: #2f6ea6;
      --brand-hover: #377fbe;
      --bull: #2d8a5e;
      --bear: #b44242;
      --gold: #c98c2a;
      --steel: #486581;
      --chart-bg: #131720;
      --chart-grid: #181f2b;
      --chart-text: #8290a4;
      --pill-active-bg: #1e2838;
      --pill-active-border: #445975;
      --pill-active-text: #e2e7ef;
    }

    :root[data-theme="light"] {
      color-scheme: light;
      --bg: #f4f6f9;
      --panel: #ffffff;
      --panel-hover: #fafbfc;
      --card-inner: #f1f4f8;
      --line: #d8e1eb;
      --line-subtle: #e4ebf3;
      --text: #1a222d;
      --text-dim: #435368;
      --muted: #6c7c91;
      --brand: #2b6194;
      --brand-hover: #22517d;
      --bull: #24754f;
      --bear: #aa3838;
      --gold: #af7319;
      --steel: #4a6785;
      --chart-bg: #ffffff;
      --chart-grid: #eef2f6;
      --chart-text: #596a80;
      --pill-active-bg: #e5edf7;
      --pill-active-border: #8ca3be;
      --pill-active-text: #1a222d;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    html, body {
      height: 100vh;
      max-height: 100vh;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 12.5px;
      line-height: 1.4;
      -webkit-font-smoothing: antialiased;
      display: flex;
      flex-direction: column;
    }

    /* ---------------- Header ---------------- */
    header {
      height: 44px;
      min-height: 44px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      background: var(--panel);
      z-index: 50;
    }
    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-logo {
      width: 20px;
      height: 20px;
      border-radius: 4px;
      background: var(--brand);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 11px;
      color: #ffffff;
    }
    .brand-title {
      font-weight: 700;
      font-size: 13.5px;
      letter-spacing: 0.04em;
      color: var(--text);
    }
    .brand-title span {
      color: var(--brand);
    }
    .version-tag {
      font-family: "Geist Mono", monospace;
      font-size: 9.5px;
      padding: 1px 5px;
      border-radius: 3px;
      background: var(--card-inner);
      color: var(--muted);
      border: 1px solid var(--line);
    }
    .header-center {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 11.5px;
      color: var(--muted);
    }
    .header-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .status-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 12px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      font-size: 11px;
      color: var(--text-dim);
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--bull);
    }

    /* Light/Dark Toggle Button */
    .theme-btn {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 4px 9px;
      border-radius: 5px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      color: var(--text);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }
    .theme-btn:hover {
      background: var(--panel-hover);
      border-color: var(--steel);
    }

    /* ---------------- Single-Screen Layout Container ---------------- */
    .app-workspace {
      flex: 1;
      min-height: 0;
      padding: 8px 12px;
      display: grid;
      grid-template-columns: 295px minmax(0, 1fr);
      gap: 10px;
      overflow: hidden;
    }

    /* ---------------- Left Sidebar (Configuration & Controls) ---------------- */
    .config-sidebar {
      display: flex;
      flex-direction: column;
      gap: 8px;
      height: 100%;
      min-height: 0;
      overflow-y: auto;
      padding-right: 2px;
    }
    .config-sidebar::-webkit-scrollbar {
      width: 4px;
    }
    .config-sidebar::-webkit-scrollbar-thumb {
      background: var(--line);
      border-radius: 4px;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .card-label {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .badge-pill {
      font-family: "Geist Mono", monospace;
      font-size: 10.5px;
      color: var(--text-dim);
      background: var(--card-inner);
      padding: 1px 6px;
      border-radius: 3px;
      border: 1px solid var(--line);
    }

    /* Inputs & Selectors */
    .select-input {
      width: 100%;
      padding: 6px 8px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--text);
      font-family: inherit;
      font-size: 12px;
      outline: none;
      transition: border-color 0.15s ease;
    }
    .select-input:focus {
      border-color: var(--brand);
    }
    .text-input {
      width: 100%;
      padding: 6px 8px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      border-radius: 5px;
      color: var(--text);
      font-family: "Geist Mono", monospace;
      font-size: 11.5px;
      outline: none;
      margin-top: 5px;
    }
    .text-input:focus {
      border-color: var(--brand);
    }
    .chips-group {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .chip-btn {
      font-family: "Geist Mono", monospace;
      font-size: 10px;
      padding: 2px 7px;
      border-radius: 3px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      color: var(--text-dim);
      cursor: pointer;
      user-select: none;
      transition: all 0.12s ease;
    }
    .chip-btn:hover {
      border-color: var(--brand);
      color: var(--text);
    }

    /* Parameters Grid */
    .param-group {
      margin-bottom: 8px;
    }
    .param-group:last-child {
      margin-bottom: 0;
    }
    .param-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 500;
    }
    .btn-pill-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 4px;
    }
    .btn-pill-grid.col-3 {
      grid-template-columns: repeat(3, 1fr);
    }
    .pill-option {
      font-family: "Geist Mono", monospace;
      font-size: 11px;
      padding: 5px 2px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      border-radius: 4px;
      color: var(--muted);
      cursor: pointer;
      text-align: center;
      user-select: none;
      transition: all 0.12s ease;
    }
    .pill-option:hover {
      color: var(--text);
      border-color: var(--steel);
    }
    .pill-option.active {
      background: var(--pill-active-bg);
      border-color: var(--pill-active-border);
      color: var(--pill-active-text);
      font-weight: 600;
    }

    /* Apply Button */
    .apply-action-btn {
      width: 100%;
      padding: 8px;
      background: var(--brand);
      color: #ffffff;
      border: 1px solid var(--brand-hover);
      border-radius: 5px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: background 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-top: 8px;
    }
    .apply-action-btn:hover {
      background: var(--brand-hover);
    }

    /* Market Metrics Card */
    .price-display {
      font-family: "Geist Mono", monospace;
      font-size: 20px;
      font-weight: 700;
      color: var(--text);
      margin: 2px 0 1px;
      letter-spacing: -0.02em;
    }
    .price-time-sub {
      font-size: 10.5px;
      color: var(--muted);
      font-family: "Geist Mono", monospace;
    }
    .metrics-grid-2x2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 8px;
    }
    .metric-item {
      padding: 6px 8px;
      background: var(--card-inner);
      border: 1px solid var(--line-subtle);
      border-radius: 5px;
    }
    .metric-item .sub-k {
      font-size: 9.5px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .metric-item b {
      display: block;
      font-family: "Geist Mono", monospace;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      margin-top: 1px;
    }
    .metric-item .sub-v {
      font-size: 10px;
      color: var(--muted);
    }
    .regime-strip {
      margin-top: 8px;
      padding: 5px 8px;
      border-radius: 4px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      text-align: center;
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.06em;
      color: var(--gold);
    }
    .spread-metric {
      display: flex;
      justify-content: space-between;
      margin-top: 7px;
      font-size: 10.5px;
      color: var(--muted);
    }
    .spread-track {
      height: 3px;
      border-radius: 3px;
      background: var(--line);
      margin-top: 4px;
      overflow: hidden;
    }
    .spread-fill {
      height: 100%;
      width: 50%;
      background: var(--brand);
      transition: width 0.25s ease;
    }

    /* Alert Notice */
    .alert-box {
      display: none;
      margin-top: 6px;
      padding: 6px 8px;
      border-radius: 4px;
      background: rgba(180, 66, 66, 0.15);
      border: 1px solid var(--bear);
      color: var(--bear);
      font-size: 10.5px;
      line-height: 1.35;
      word-break: break-word;
    }

    /* ---------------- Right Panel (Chart Area) ---------------- */
    .chart-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .chart-topbar {
      height: 36px;
      min-height: 36px;
      padding: 0 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--panel);
    }
    .chart-title-left {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text);
    }
    .chart-title-left .pair-callout {
      font-family: "Geist Mono", monospace;
      color: var(--brand);
      font-size: 12px;
    }
    .chart-topbar-right {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .progress-box {
      display: flex;
      align-items: center;
      gap: 7px;
      font-family: "Geist Mono", monospace;
      font-size: 11px;
      color: var(--text-dim);
    }
    .progress-bar-small {
      width: 75px;
      height: 4px;
      background: var(--card-inner);
      border-radius: 3px;
      border: 1px solid var(--line);
      overflow: hidden;
    }
    .progress-bar-small span {
      display: block;
      height: 100%;
      width: 0%;
      background: var(--bull);
      transition: width 0.2s ease;
    }
    .resample-btn {
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--card-inner);
      border: 1px solid var(--line);
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.12s ease;
    }
    .resample-btn:hover {
      border-color: var(--steel);
      color: var(--text);
    }
    .chart-canvas-wrapper {
      flex: 1;
      min-height: 0;
      width: 100%;
      position: relative;
    }

    /* ---------------- Compact Footer ---------------- */
    footer {
      height: 22px;
      min-height: 22px;
      padding: 0 16px;
      border-top: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 10.5px;
      color: var(--muted);
      background: var(--panel);
    }
    footer a {
      color: var(--muted);
      text-decoration: none;
    }

    @media (max-width: 860px) {
      html, body {
        height: auto;
        max-height: none;
        overflow: auto;
      }
      .app-workspace {
        grid-template-columns: 1fr;
        height: auto;
        overflow: visible;
      }
      .chart-canvas-wrapper {
        height: 460px;
      }
    }
  </style>
</head>
<body>

  <!-- Top Bar -->
  <header>
    <div class="brand-wrap">
      <div class="brand-logo">◈</div>
      <div class="brand-title">NULL<span>HORIZON</span></div>
      <span class="version-tag">v1.1</span>
    </div>
    <div class="header-center">
      <span>Binance Spot Kline Pipeline</span>
      <span>·</span>
      <span id="telemetryDevice">Checking CUDA...</span>
    </div>
    <div class="header-right">
      <div class="status-indicator">
        <i class="pulse-dot"></i>
        <span id="headerStatusText">Connecting...</span>
      </div>
      <button type="button" class="theme-btn" id="themeBtn" onclick="toggleTheme()">
        <span id="themeIcon">☀️</span>
        <span id="themeLabel">Light</span>
      </button>
    </div>
  </header>

  <!-- Workspace: Left Configuration Menu + Right Chart Canvas -->
  <div class="app-workspace">

    <!-- Configuration Menu (LEFT SIDE) -->
    <aside class="config-sidebar">

      <!-- Card 1: Currency (Trading Pair) -->
      <div class="card">
        <div class="card-label">
          <span>Currency / Pair</span>
          <span class="badge-pill" id="pairBadge">{{ symbol }}</span>
        </div>
        <select id="currencyDropdown" class="select-input" onchange="onDropdownChange()">
          <optgroup label="Spot Cryptocurrencies">
            <option value="BTCUSDT">BTC / USDT (Bitcoin)</option>
            <option value="ETHUSDT">ETH / USDT (Ethereum)</option>
            <option value="SOLUSDT">SOL / USDT (Solana)</option>
            <option value="BNBUSDT">BNB / USDT (Binance Coin)</option>
            <option value="POLUSDT">POL / USDT (Polygon)</option>
            <option value="XRPUSDT">XRP / USDT (Ripple)</option>
            <option value="DOGEUSDT">DOGE / USDT (Dogecoin)</option>
            <option value="ADAUSDT">ADA / USDT (Cardano)</option>
            <option value="AVAXUSDT">AVAX / USDT (Avalanche)</option>
            <option value="NEARUSDT">NEAR / USDT (NEAR Protocol)</option>
            <option value="SUIUSDT">SUI / USDT (Sui)</option>
            <option value="LINKUSDT">LINK / USDT (Chainlink)</option>
          </optgroup>
          <optgroup label="Custom Asset">
            <option value="CUSTOM">Custom pair...</option>
          </optgroup>
        </select>
        <input type="text" id="customInput" class="text-input" placeholder="e.g. PEPEUSDT" style="display:none;" onkeydown="if(event.key==='Enter')applyParameters()">
        <div class="chips-group">
          <span class="chip-btn" onclick="quickPair('BTCUSDT')">BTC</span>
          <span class="chip-btn" onclick="quickPair('ETHUSDT')">ETH</span>
          <span class="chip-btn" onclick="quickPair('SOLUSDT')">SOL</span>
          <span class="chip-btn" onclick="quickPair('POLUSDT')">POL</span>
          <span class="chip-btn" onclick="quickPair('DOGEUSDT')">DOGE</span>
          <span class="chip-btn" onclick="quickPair('SUIUSDT')">SUI</span>
        </div>
      </div>

      <!-- Card 2: Horizon, Candle Frequency & Paths -->
      <div class="card">
        <div class="card-label">
          <span>Simulation Horizon</span>
          <span class="badge-pill" id="horizonBadge">{{ horizon }} bars</span>
        </div>

        <!-- Frequency -->
        <div class="param-group">
          <div class="param-head">
            <span>Candle Frequency</span>
            <span id="freqBadge">{{ interval }}</span>
          </div>
          <div class="btn-pill-grid" id="freqGrid">
            <div class="pill-option" data-freq="15m" onclick="setFreq('15m')">15m</div>
            <div class="pill-option active" data-freq="1h" onclick="setFreq('1h')">1h</div>
            <div class="pill-option" data-freq="4h" onclick="setFreq('4h')">4h</div>
            <div class="pill-option" data-freq="1d" onclick="setFreq('1d')">1d</div>
            <div class="pill-option" data-freq="1w" onclick="setFreq('1w')">1w</div>
          </div>
        </div>

        <!-- Horizon -->
        <div class="param-group">
          <div class="param-head">
            <span>Forecast Horizon</span>
            <span id="barsBadge">{{ horizon }} bars</span>
          </div>
          <div class="btn-pill-grid" id="horizonGrid">
            <div class="pill-option" data-hz="24" onclick="setHorizon(24)">24</div>
            <div class="pill-option" data-hz="48" onclick="setHorizon(48)">48</div>
            <div class="pill-option active" data-hz="72" onclick="setHorizon(72)">72</div>
            <div class="pill-option" data-hz="120" onclick="setHorizon(120)">120</div>
            <div class="pill-option" data-hz="168" onclick="setHorizon(168)">168</div>
          </div>
        </div>

        <!-- Paths -->
        <div class="param-group">
          <div class="param-head">
            <span>Monte Carlo Paths</span>
            <span id="pathsBadge">{{ paths }}</span>
          </div>
          <div class="btn-pill-grid col-3" id="pathsGrid">
            <div class="pill-option" data-paths="50" onclick="setPaths(50)">50</div>
            <div class="pill-option active" data-paths="100" onclick="setPaths(100)">100</div>
            <div class="pill-option" data-paths="200" onclick="setPaths(200)">200</div>
          </div>
        </div>

        <!-- Apply Button -->
        <button type="button" class="apply-action-btn" id="applyBtn" onclick="applyParameters()">
          <span>⚡ Apply & Run Forecast</span>
        </button>
        <div id="alertBox" class="alert-box"></div>
      </div>

      <!-- Card 3: Metrics & Regime -->
      <div class="card">
        <div class="card-label">
          <span>Market Statistics</span>
          <span class="badge-pill" id="regimeBadge">INITIALIZING</span>
        </div>
        <div class="price-display" id="lastPrice">—</div>
        <div class="price-time-sub" id="lastCandleTime">Awaiting stream...</div>

        <div class="metrics-grid-2x2">
          <div class="metric-item">
            <div class="sub-k" id="p50Title">Median (P50)</div>
            <b id="p50Val">—</b>
            <div class="sub-v" id="p50Chg">—</div>
          </div>
          <div class="metric-item">
            <div class="sub-k">Upward Paths</div>
            <b id="upPctVal">—</b>
            <div class="sub-v" id="upCountVal">—</div>
          </div>
          <div class="metric-item">
            <div class="sub-k">P10 Lower</div>
            <b id="p10Val">—</b>
            <div class="sub-v">10th quantile</div>
          </div>
          <div class="metric-item">
            <div class="sub-k">P90 Upper</div>
            <b id="p90Val">—</b>
            <div class="sub-v">90th quantile</div>
          </div>
        </div>

        <div class="spread-metric">
          <span>Spread (P90 − P10)</span>
          <span id="spreadVal" style="font-family:'Geist Mono',monospace;font-weight:600;">—</span>
        </div>
        <div class="spread-track">
          <div class="spread-fill" id="spreadBarFill"></div>
        </div>
      </div>

    </aside>

    <!-- Right Chart Panel -->
    <main class="chart-panel">
      <div class="chart-topbar">
        <div class="chart-title-left">
          <span>Candles & Autoregressive Paths</span>
          <span class="pair-callout" id="activePairCallout">{{ symbol }} · {{ interval }}</span>
        </div>
        <div class="chart-topbar-right">
          <div class="progress-box">
            <span id="progressLabel">0/{{ paths }}</span>
            <div class="progress-bar-small">
              <span id="progressBarSpan"></span>
            </div>
          </div>
          <button type="button" class="resample-btn" onclick="applyParameters()">↻ Re-Sample</button>
        </div>
      </div>
      <div id="chartContainer" class="chart-canvas-wrapper"></div>
    </main>

  </div>

  <!-- Bottom Bar -->
  <footer>
    <span>NullHorizon v1.1 · Autoregressive Candlestick Transformer</span>
    <span>Context: {{ context }} bars · Model: Kronos-base (102.3M)</span>
  </footer>

  <!-- App Script -->
  <script>
    let currentTheme = localStorage.getItem('nh_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);

    let config = {
      symbol: "{{ symbol }}",
      interval: "{{ interval }}",
      horizon: {{ horizon }},
      paths: {{ paths }}
    };

    function fmtPrice(val) {
      if (val === null || val === undefined || isNaN(val)) return "—";
      const num = Number(val);
      if (num >= 10000) return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      if (num >= 1) return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 3 });
      if (num >= 0.01) return num.toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });
      return num.toLocaleString('en-US', { minimumFractionDigits: 6, maximumFractionDigits: 8 });
    }

    // Chart Configuration
    const chartContainer = document.getElementById('chartContainer');
    const isDark = currentTheme === 'dark';

    const chart = LightweightCharts.createChart(chartContainer, {
      layout: {
        background: { color: isDark ? '#131720' : '#ffffff' },
        textColor: isDark ? '#8290a4' : '#596a80',
        fontFamily: 'Geist Mono, Inter, -apple-system, monospace',
        fontSize: 11
      },
      grid: {
        vertLines: { color: isDark ? '#181f2b' : '#eef2f6' },
        horzLines: { color: isDark ? '#181f2b' : '#eef2f6' }
      },
      rightPriceScale: {
        borderColor: isDark ? '#202634' : '#d8e1eb',
        scaleMargins: { top: 0.1, bottom: 0.1 }
      },
      timeScale: {
        borderColor: isDark ? '#202634' : '#d8e1eb',
        timeVisible: true,
        secondsVisible: false
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { color: 'rgba(74, 103, 133, 0.4)', style: 3 },
        horzLine: { color: 'rgba(74, 103, 133, 0.4)', style: 3 }
      },
      handleScroll: true,
      handleScale: true
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#2d8a5e',
      downColor: '#b44242',
      borderUpColor: '#2d8a5e',
      borderDownColor: '#b44242',
      wickUpColor: '#2d8a5e',
      wickDownColor: '#b44242'
    });

    const medSeries = chart.addLineSeries({
      color: '#c98c2a',
      lineWidth: 2,
      title: 'P50 Median',
      priceLineVisible: false
    });

    const lowSeries = chart.addLineSeries({
      color: 'rgba(72, 101, 129, 0.85)',
      lineWidth: 1,
      lineStyle: 2,
      title: 'P10',
      priceLineVisible: false
    });

    const highSeries = chart.addLineSeries({
      color: 'rgba(72, 101, 129, 0.85)',
      lineWidth: 1,
      lineStyle: 2,
      title: 'P90',
      priceLineVisible: false
    });

    let pathSeries = [];
    let lastCycle = -1;
    let lastPathCount = 0;

    function resizeChart() {
      chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight
      });
    }
    window.addEventListener('resize', resizeChart);
    setTimeout(resizeChart, 50);

    // Theme Toggle Functionality
    function updateThemeUI() {
      const isL = currentTheme === 'light';
      document.getElementById('themeIcon').textContent = isL ? '🌙' : '☀️';
      document.getElementById('themeLabel').textContent = isL ? 'Dark' : 'Light';
      
      chart.applyOptions({
        layout: {
          background: { color: isL ? '#ffffff' : '#131720' },
          textColor: isL ? '#596a80' : '#8290a4'
        },
        grid: {
          vertLines: { color: isL ? '#eef2f6' : '#181f2b' },
          horzLines: { color: isL ? '#eef2f6' : '#181f2b' }
        },
        rightPriceScale: { borderColor: isL ? '#d8e1eb' : '#202634' },
        timeScale: { borderColor: isL ? '#d8e1eb' : '#202634' }
      });
    }

    function toggleTheme() {
      currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', currentTheme);
      localStorage.setItem('nh_theme', currentTheme);
      updateThemeUI();
    }
    updateThemeUI();

    // Configuration Handlers
    function onDropdownChange() {
      const select = document.getElementById('currencyDropdown');
      const customInput = document.getElementById('customInput');
      if (select.value === 'CUSTOM') {
        customInput.style.display = 'block';
        customInput.focus();
      } else {
        customInput.style.display = 'none';
        config.symbol = select.value;
        document.getElementById('pairBadge').textContent = config.symbol;
      }
    }

    function quickPair(pair) {
      config.symbol = pair;
      document.getElementById('currencyDropdown').value = pair;
      document.getElementById('customInput').style.display = 'none';
      document.getElementById('pairBadge').textContent = pair;
      applyParameters();
    }

    function setFreq(itv) {
      config.interval = itv;
      document.querySelectorAll('#freqGrid .pill-option').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-freq') === itv);
      });
      document.getElementById('freqBadge').textContent = itv;
    }

    function setHorizon(hz) {
      config.horizon = hz;
      document.querySelectorAll('#horizonGrid .pill-option').forEach(el => {
        el.classList.toggle('active', parseInt(el.getAttribute('data-hz')) === hz);
      });
      document.getElementById('horizonBadge').textContent = hz + ' bars';
      document.getElementById('barsBadge').textContent = hz + ' bars';
      document.getElementById('p50Title').textContent = 'Median (+' + hz + ')';
    }

    function setPaths(paths) {
      config.paths = paths;
      document.querySelectorAll('#pathsGrid .pill-option').forEach(el => {
        el.classList.toggle('active', parseInt(el.getAttribute('data-paths')) === paths);
      });
      document.getElementById('pathsBadge').textContent = paths;
    }

    async function applyParameters() {
      const select = document.getElementById('currencyDropdown');
      const customInput = document.getElementById('customInput');
      if (select.value === 'CUSTOM') {
        const val = customInput.value.trim().toUpperCase();
        if (val) config.symbol = val;
      } else {
        config.symbol = select.value;
      }

      document.getElementById('pairBadge').textContent = config.symbol;
      document.getElementById('activePairCallout').textContent = config.symbol + ' · ' + config.interval;

      const applyBtn = document.getElementById('applyBtn');
      applyBtn.disabled = true;
      applyBtn.style.opacity = '0.7';

      try {
        const res = await fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.ok) hideAlert();
      } catch (err) {
        showAlert('Request failed: ' + err.message);
      } finally {
        applyBtn.disabled = false;
        applyBtn.style.opacity = '1';
      }
    }

    function showAlert(msg) {
      const el = document.getElementById('alertBox');
      el.textContent = msg;
      el.style.display = 'block';
    }

    function hideAlert() {
      const el = document.getElementById('alertBox');
      el.style.display = 'none';
      el.textContent = '';
    }

    // Polling & Progressive Rendering
    async function pollState() {
      try {
        const res = await fetch('/api/state', { cache: 'no-store' });
        const d = await res.json();

        document.getElementById('headerStatusText').textContent = d.status;
        document.getElementById('telemetryDevice').textContent = d.device || 'CUDA';

        if (d.error) {
          showAlert(d.error);
        } else {
          hideAlert();
        }

        document.getElementById('progressLabel').textContent = `${d.completed_paths}/${d.target_paths}`;
        document.getElementById('progressBarSpan').style.width = d.progress + '%';

        if (!d.candles || !d.candles.length) return;

        if (d.cycle !== lastCycle) {
          candleSeries.setData(d.candles);
          lastCycle = d.cycle;
          lastPathCount = 0;
          pathSeries.forEach(s => chart.removeSeries(s));
          pathSeries = [];
          medSeries.setData([]);
          lowSeries.setData([]);
          highSeries.setData([]);
          chart.timeScale().fitContent();
        }

        if (d.completed_paths > lastPathCount) {
          const isL = currentTheme === 'light';
          for (let i = lastPathCount; i < d.paths.length; i++) {
            const isHighlight = i % 10 === 0;
            const s = chart.addLineSeries({
              color: isHighlight 
                ? (isL ? 'rgba(43, 97, 148, 0.4)' : 'rgba(74, 103, 133, 0.45)')
                : (isL ? 'rgba(100, 116, 139, 0.15)' : 'rgba(130, 144, 164, 0.12)'),
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
              crosshairMarkerVisible: false
            });
            s.setData(d.paths[i]);
            pathSeries.push(s);
          }
          lastPathCount = d.completed_paths;
        }

        const s = d.stats;
        if (s && s.last) {
          document.getElementById('lastPrice').textContent = '$' + fmtPrice(s.last);
          document.getElementById('lastCandleTime').textContent = 'Last: ' + new Date(s.last_candle_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          document.getElementById('p50Val').textContent = '$' + fmtPrice(s.p50);
          document.getElementById('p10Val').textContent = '$' + fmtPrice(s.p10);
          document.getElementById('p90Val').textContent = '$' + fmtPrice(s.p90);

          const chg = s.median_change_pct || 0;
          const chgEl = document.getElementById('p50Chg');
          chgEl.textContent = (chg > 0 ? '+' : '') + chg + '% vs close';
          chgEl.style.color = chg > 0 ? 'var(--bull)' : (chg < 0 ? 'var(--bear)' : 'var(--muted)');

          document.getElementById('upPctVal').textContent = (s.up_pct || 0) + '%';
          document.getElementById('upCountVal').textContent = `${s.partial_paths || 0} futures`;

          const spread = (s.p90 || 0) - (s.p10 || 0);
          document.getElementById('spreadVal').textContent = '$' + fmtPrice(spread);

          const regimeEl = document.getElementById('regimeBadge');
          regimeEl.textContent = s.regime || 'CALCULATING';
          if (s.regime === 'BULLISH') {
            regimeEl.style.color = 'var(--bull)';
            regimeEl.style.borderColor = 'var(--bull)';
          } else if (s.regime === 'BEARISH') {
            regimeEl.style.color = 'var(--bear)';
            regimeEl.style.borderColor = 'var(--bear)';
          } else {
            regimeEl.style.color = 'var(--gold)';
            regimeEl.style.borderColor = 'var(--gold)';
          }

          document.getElementById('spreadBarFill').style.width = Math.max(5, Math.min(100, s.up_pct || 50)) + '%';

          if (s.forecast_times && s.partial_median && s.forecast_times.length === s.partial_median.length) {
            const anchor = { time: d.candles[d.candles.length - 1].time, value: s.last };
            medSeries.setData([anchor, ...s.forecast_times.map((t, idx) => ({ time: t, value: s.partial_median[idx] }))]);
            lowSeries.setData([anchor, ...s.forecast_times.map((t, idx) => ({ time: t, value: s.partial_p10[idx] }))]);
            highSeries.setData([anchor, ...s.forecast_times.map((t, idx) => ({ time: t, value: s.partial_p90[idx] }))]);
          }
        }
      } catch (err) {}
    }

    setInterval(pollState, 350);
    pollState();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    with lock:
        return render_template_string(
            PAGE_TEMPLATE,
            symbol=state["symbol"],
            interval=state["interval"],
            horizon=state["horizon"],
            paths=state["target_paths"],
            context=state["context"],
        )


def main():
    global predictor, device_label
    parser = argparse.ArgumentParser(description="NullHorizon v1.1: Probabilistic Market Forecaster")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port number (default: {DEFAULT_PORT})")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Initial trading pair (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help=f"Candle interval (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help=f"Forecast horizon (default: {DEFAULT_HORIZON})")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS, help=f"Monte Carlo paths (default: {DEFAULT_PATHS})")
    parser.add_argument("--model", default=MODEL_ID, help=f"Model path or HuggingFace ID (default: {MODEL_ID})")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID, help=f"Tokenizer path or HuggingFace ID (default: {TOKENIZER_ID})")
    args = parser.parse_args()

    model_source = args.model
    tokenizer_source = args.tokenizer
    model_name_display = "Kronos-base (Local)" if os.path.isdir(model_source) else model_source

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        device_label = f"CUDA ({gpu_name})"
    else:
        device_label = "CPU"

    with lock:
        state["symbol"] = args.symbol.strip().upper()
        state["interval"] = args.interval
        state["horizon"] = args.horizon
        state["target_paths"] = args.paths
        state["model"] = model_name_display
        state["device"] = device_label

    logger.info(f"Compute accelerator: {device_label}")
    logger.info(f"Loading foundation model from: {model_source}")
    logger.info(f"Loading tokenizer from: {tokenizer_source}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_source)
    model = Kronos.from_pretrained(model_source)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=CONTEXT_LEN)

    logger.info(f"Model ready. Starting NullHorizon v1.1 at http://{args.host}:{args.port}")

    threading.Thread(target=background_loop, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
