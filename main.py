import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from scipy.signal import find_peaks
import plotly.graph_objects as go
# --- LSTM Forecast Integration ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import time
from typing import Tuple
import joblib

from investing import (
    ensure_investing_tables,
    upsert_companies,
    upsert_fundamentals_ttm,
    upsert_market_snapshots,
    upsert_scores,
    save_screen_results,
    load_last_screen_results,
    compute_investing_scores,
    persist_investing_run,
    render_investing_analysis,   # <— NEU

)

# ==== DB Imports ====
import sqlite3
import json
DB_PATH = "market_dashboard.db"

# --- Local DCF/Valuation helpers (decoupled from investing.py) ---

def _safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def estimate_dcf_fair_value(tk: str, r: float = 0.10, g_years: int = 5,
                            g_initial: float | None = None, g_terminal: float = 0.02) -> dict:
    """Kompakte FCFF-DCF-Schätzung auf Basis yfinance-Daten.
    FCFF ≈ Operating CF - CapEx; Diskontierung mit r; Terminal via Gordon; Equity = EV - NetDebt; FV/share = Equity / shares.
    Liefert: fair_value, deviation (FV/Preis - 1), inputs, warnings.
    """
    t = yf.Ticker(tk)
    info = getattr(t, 'info', {}) or {}
    cf  = getattr(t, 'cashflow', None)
    bs  = getattr(t, 'balance_sheet', None)

    # FCFF-Historie
    try:
        op_cf = pd.to_numeric(cf.loc['Operating Cash Flow'], errors='coerce').dropna().sort_index()
    except Exception:
        op_cf = pd.Series(dtype='float64')
    try:
        capex = pd.to_numeric(cf.loc['Capital Expenditure'], errors='coerce').dropna().sort_index() * -1.0
    except Exception:
        capex = pd.Series(dtype='float64')
    fcff_hist = (op_cf.add(capex, fill_value=0)).dropna()
    if fcff_hist.empty:
        return {'fair_value': np.nan, 'inputs': {}, 'warnings': ['Keine FCFF-Daten']}

    base = float(fcff_hist.iloc[-1])
    if len(fcff_hist) >= 3:
        first = float(fcff_hist.iloc[-3]); last = float(fcff_hist.iloc[-1]); years = 2
        try:
            g_calc = (last/first)**(1/years) - 1.0 if first > 0 and years > 0 else np.nan
        except Exception:
            g_calc = np.nan
    else:
        g_calc = np.nan
    g0 = g_initial if g_initial is not None else (0.06 if pd.isna(g_calc) else float(np.clip(g_calc, -0.05, 0.20)))

    shares = _safe_num(info.get('sharesOutstanding'))
    price  = _safe_num(info.get('currentPrice'))
    debt, cash = np.nan, np.nan
    if bs is not None and not bs.empty:
        if 'Total Debt' in bs.index:
            debt = _safe_num(pd.to_numeric(bs.loc['Total Debt'], errors='coerce').dropna().sort_index().iloc[-1])
        if 'Cash And Cash Equivalents' in bs.index:
            cash = _safe_num(pd.to_numeric(bs.loc['Cash And Cash Equivalents'], errors='coerce').dropna().sort_index().iloc[-1])
    net_debt = (0 if pd.isna(debt) else debt) - (0 if pd.isna(cash) else cash)

    if pd.isna(shares) or shares <= 0:
        return {'fair_value': np.nan, 'inputs': {}, 'warnings': ['Keine SharesOutstanding-Daten']}

    r = float(max(0.05, min(0.18, r)))
    g_terminal = float(max(-0.02, min(0.03, g_terminal)))
    g_years = int(max(3, min(7, g_years)))

    # Forecast FCFF
    fcff_fore = []
    cur = base
    for _ in range(g_years):
        cur = cur * (1.0 + g0)
        fcff_fore.append(cur)

    # Terminal
    tv = fcff_fore[-1] * (1.0 + g_terminal) / max(1e-9, (r - g_terminal))

    def _pv(x, t):
        return x / ((1.0 + r) ** t)

    pv_fcffs = sum(_pv(x, i) for i, x in enumerate(fcff_fore, start=1))
    pv_tv = _pv(tv, g_years)
    ev = pv_fcffs + pv_tv
    equity = ev - net_debt
    fv_per_share = equity / shares

    deviation = np.nan if pd.isna(price) else (fv_per_share / price - 1.0)
    return {
        'fair_value': float(fv_per_share),
        'deviation': None if pd.isna(deviation) else float(deviation),
        'inputs': {
            'base_fcff': float(base), 'g_initial': float(g0), 'g_years': int(g_years),
            'g_terminal': float(g_terminal), 'discount': float(r), 'net_debt': float(net_debt),
            'shares': float(shares), 'price': None if pd.isna(price) else float(price)
        },
        'warnings': [],
    }

def valuation_status_from_deviation(dev: float | None) -> tuple[str, str]:
    if dev is None or pd.isna(dev):
        return ("n/a", "#999999")
    if dev >= 0.20:
        return ("Unterbewertet", "#2ecc71")
    if -0.20 <= dev < 0.20:
        return ("Fair bewertet", "#f1c40f")
    return ("Überbewertet", "#e74c3c")

# Tabellen für Investing-Modul sicherstellen (idempotent) und Pfad vereinheitlichen
from investing import set_db_path as _invest_set_db_path
_invest_set_db_path(DB_PATH)
ensure_investing_tables(DB_PATH)


# ---- Unified DB helpers ----
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        # Improve concurrency and durability for Streamlit + SQLite
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=30000000000;")  # 30GB cap; OS will limit as needed
    except Exception:
        pass
    return conn

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        # Base tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lstm_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date_generated TEXT,
                forecast_date TEXT,
                forecast REAL,
                upper_band REAL,
                lower_band REAL,
                model_params_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS confluence_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                level REAL,
                score INTEGER,
                low REAL,
                high REAL,
                generated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_results (
                trade_date TEXT,
                product    TEXT,
                pnl        REAL,
                PRIMARY KEY(trade_date, product)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                ticker TEXT,
                snap_date TEXT,
                price REAL,
                market_cap REAL,
                ev REAL,
                total_return_3m REAL,
                total_return_6m REAL,
                total_return_12m REAL,
                above_ma200 REAL,
                PRIMARY KEY(ticker, snap_date)
            )
            """
        )

        # --- Deduplicate existing data before creating UNIQUE indices ---
        # Keep the earliest row per natural key and delete the rest
        cur.execute(
            """
            DELETE FROM historical_prices
            WHERE id NOT IN (
                SELECT MIN(id) FROM historical_prices GROUP BY ticker, date
            )
            """
        )
        cur.execute(
            """
            DELETE FROM lstm_forecasts
            WHERE id NOT IN (
                SELECT MIN(id) FROM lstm_forecasts GROUP BY ticker, forecast_date
            )
            """
        )
        cur.execute(
            """
            DELETE FROM confluence_zones
            WHERE id NOT IN (
                SELECT MIN(id) FROM confluence_zones GROUP BY ticker, level, generated_at
            )
            """
        )

        # Create UNIQUE indices with a safety retry in case duplicates appear concurrently
        try:
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_unique
                ON historical_prices(ticker, date)
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_forecasts_unique
                ON lstm_forecasts(ticker, forecast_date)
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_zones_unique
                ON confluence_zones(ticker, level, generated_at)
                """
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Last‑chance dedup, then retry index creation
            cur.execute(
                """
                DELETE FROM historical_prices
                WHERE id NOT IN (
                    SELECT MIN(id) FROM historical_prices GROUP BY ticker, date
                )
                """
            )
            cur.execute(
                """
                DELETE FROM lstm_forecasts
                WHERE id NOT IN (
                    SELECT MIN(id) FROM lstm_forecasts GROUP BY ticker, forecast_date
                )
                """
            )
            cur.execute(
                """
                DELETE FROM confluence_zones
                WHERE id NOT IN (
                    SELECT MIN(id) FROM confluence_zones GROUP BY ticker, level, generated_at
                )
                """
            )
            conn.commit()
            # Retry creating indices
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_unique
                ON historical_prices(ticker, date)
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_forecasts_unique
                ON lstm_forecasts(ticker, forecast_date)
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_zones_unique
                ON confluence_zones(ticker, level, generated_at)
                """
            )
            conn.commit()

# Initialize DB on import
init_db()
# only new prices will be writen into DB
def _get_latest_price_date(ticker: str):
    """Return latest stored date for ticker in historical_prices, or None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM historical_prices WHERE ticker = ?", (ticker,))
        row = cur.fetchone()
        if row and row[0]:
            try:
                return pd.to_datetime(row[0])
            except Exception as e:
                # Log once to Streamlit if available; otherwise ignore
                try:
                    st.warning(f"Konnte letztes Preisdaten-Datum nicht parsen: {e}")
                except Exception:
                    pass
                return None
        return None

def upsert_forecast(ticker, forecast_df, params_dict):
    params_json = json.dumps(params_dict)
    with get_connection() as conn:
        cur = conn.cursor()
        for _, row in forecast_df.iterrows():
            cur.execute(
                """
                INSERT INTO lstm_forecasts (ticker, date_generated, forecast_date, forecast, upper_band, lower_band, model_params_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, forecast_date) DO UPDATE SET
                    date_generated = excluded.date_generated,
                    forecast = excluded.forecast,
                    upper_band = excluded.upper_band,
                    lower_band = excluded.lower_band,
                    model_params_json = excluded.model_params_json
                """,
                (
                    ticker,
                    pd.Timestamp.today().strftime("%Y-%m-%d"),
                    row['Date'].strftime("%Y-%m-%d"),
                    float(row['Forecast']),
                    float(row['Upper']),
                    float(row['Lower']),
                    params_json,
                )
            )
        conn.commit()

# --- Trading-Log: Upsert und Laden (verwenden unified helper) ---
def upsert_trading_result(trade_date, product, pnl):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO trading_results (trade_date, product, pnl)
            VALUES (?, ?, ?)
            ON CONFLICT(trade_date, product) DO UPDATE SET pnl = excluded.pnl
            """,
            (trade_date.strftime("%Y-%m-%d"), product, pnl)
        )
        conn.commit()

def load_trading_results():
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT trade_date, product, pnl FROM trading_results ORDER BY trade_date",
            conn
        )
    return df

def upsert_zones(ticker, zones_list):
    with get_connection() as conn:
        cur = conn.cursor()
        generated_at = pd.Timestamp.today().strftime("%Y-%m-%d")
        for zone in zones_list:
            cur.execute(
                """
                INSERT INTO confluence_zones (ticker, level, score, low, high, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, level, generated_at) DO UPDATE SET
                    score = excluded.score,
                    low   = excluded.low,
                    high  = excluded.high
                """,
                (
                    ticker,
                    float(zone['level']),
                    int(zone['score']),
                    float(zone['low']),
                    float(zone['high']),
                    generated_at,
                )
            )
        conn.commit()

def upsert_prices(ticker, df, only_new: bool = True):
    """Upsert OHLCV rows. If only_new, insert rows with date > latest stored date."""
    if df is None or df.empty:
        return
    local_df = df.copy()
    local_df.index = pd.to_datetime(local_df.index)

    if only_new:
        latest = _get_latest_price_date(ticker)
        if latest is not None:
            local_df = local_df[local_df.index > latest]
            if local_df.empty:
                return

    with get_connection() as conn:
        cur = conn.cursor()
        rows = [
            (
                ticker,
                idx.strftime("%Y-%m-%d"),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                float(row.get('Volume', 0)),
            )
            for idx, row in local_df.iterrows()
        ]
        if not rows:
            return
        cur.executemany(
            """
            INSERT INTO historical_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low  = excluded.low,
                close= excluded.close,
                volume = excluded.volume
            """,
            rows
        )
        conn.commit()


# --- Market Snapshots Upsert Helper ---
def save_market_snapshots(df: pd.DataFrame):
    """
    Upsert market snapshot rows into the local DB.
    Expected columns:
      ticker, snap_date, [price], [market_cap], [ev], [total_return_3m], [total_return_6m], [total_return_12m], [above_ma200]
    """
    if df is None or df.empty:
        return
    cols = ["ticker","snap_date","price","market_cap","ev","total_return_3m","total_return_6m","total_return_12m","above_ma200"]
    local = df.copy()
    # Ensure all required columns exist
    for c in cols:
        if c not in local.columns:
            local[c] = np.nan
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO market_snapshots (
                ticker, snap_date, price, market_cap, ev,
                total_return_3m, total_return_6m, total_return_12m, above_ma200
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, snap_date) DO UPDATE SET
                price = excluded.price,
                market_cap = excluded.market_cap,
                ev = excluded.ev,
                total_return_3m = excluded.total_return_3m,
                total_return_6m = excluded.total_return_6m,
                total_return_12m = excluded.total_return_12m,
                above_ma200 = excluded.above_ma200
            """,
            [
                (
                    str(row["ticker"]),
                    str(row["snap_date"]),
                    None if pd.isna(row["price"]) else float(row["price"]),
                    None if pd.isna(row["market_cap"]) else float(row["market_cap"]),
                    None if pd.isna(row["ev"]) else float(row["ev"]),
                    None if pd.isna(row["total_return_3m"]) else float(row["total_return_3m"]),
                    None if pd.isna(row["total_return_6m"]) else float(row["total_return_6m"]),
                    None if pd.isna(row["total_return_12m"]) else float(row["total_return_12m"]),
                    None if pd.isna(row["above_ma200"]) else float(row["above_ma200"]),
                )
                for _, row in local.iterrows()
            ]
        )
        conn.commit()

# === Investing Tab (Stock Screener) ===
# -- Small utilities reused by the screener --

def _cagr(first: float, last: float, years: int):
    import numpy as _np
    try:
        if first is None or last is None or years <= 0 or first <= 0:
            return _np.nan
        return (last / first) ** (1 / years) - 1
    except Exception:
        return _np.nan

def _clean_ratio(numer, denom):
    import pandas as _pd
    try:
        if denom is None or denom == 0 or _pd.isna(numer) or _pd.isna(denom):
            return _np.nan
        return float(numer) / float(denom)
    except Exception:
        return _np.nan


def _get_row(df, row_name):
    try:
        s = df.loc[row_name]
        return s.sort_index()
    except Exception:
        import pandas as _pd
        return _pd.Series(dtype="float64")


def _compute_screener_metrics(tk: str):
    """Compute a compact set of quality and price metrics for a ticker using yfinance."""
    import yfinance as _yf
    import numpy as _np
    import pandas as _pd

    t = _yf.Ticker(tk)
    info = t.info
    fin = t.financials
    bs = t.balance_sheet
    cf = t.cashflow

    # pull series
    revenue = _get_row(fin, "Total Revenue")
    net_income = _get_row(fin, "Net Income")
    gross_profit = _get_row(fin, "Gross Profit")
    ebit = _get_row(fin, "Ebit") if "Ebit" in fin.index else _get_row(fin, "EBIT")
    interest_exp = _get_row(fin, "Interest Expense")
    total_equity = _get_row(bs, "Total Stockholder Equity")
    total_debt = _get_row(bs, "Total Debt")
    cash = _get_row(bs, "Cash And Cash Equivalents")
    op_cf = _get_row(cf, "Operating Cash Flow")
    capex = _get_row(cf, "Capital Expenditure") * -1
    fcf = op_cf - capex

    # derive
    def _first_last(series):
        s = series.dropna().sort_index()
        if len(s) < 2:
            return (_np.nan, _np.nan, 0)
        return float(s.iloc[0]), float(s.iloc[-1]), max(1, len(s) - 1)

    sales_cagr = _np.nan
    profit_cagr = _np.nan
    r1, r2, ry = _first_last(revenue.tail(5))
    n1, n2, ny = _first_last(net_income.tail(5))
    if ry >= 2:
        sales_cagr = _cagr(r1, r2, ry)
    if ny >= 2:
        profit_cagr = _cagr(n1, n2, ny)

    try:
        roe = _clean_ratio(float(net_income.sort_index().iloc[-1]), float(total_equity.sort_index().iloc[-1]))
    except Exception:
        roe = _np.nan

    try:
        dte = _clean_ratio(float(total_debt.sort_index().iloc[-1]), float(total_equity.sort_index().iloc[-1]))
    except Exception:
        dte = _np.nan

    try:
        gm_latest = float((gross_profit / revenue).dropna().sort_index().iloc[-1])
        gm_std = float((gross_profit / revenue).dropna().std())
    except Exception:
        gm_latest, gm_std = _np.nan, _np.nan

    try:
        icov = _clean_ratio(float(ebit.sort_index().iloc[-1]), abs(float(interest_exp.sort_index().iloc[-1])) if len(interest_exp) else _np.nan)
    except Exception:
        icov = _np.nan

    # price ratios
    pe = info.get("trailingPE", _np.nan)
    price = info.get("currentPrice", _np.nan)
    mcap = info.get("marketCap", _np.nan)
    shares = info.get("sharesOutstanding", _np.nan)
    try:
        fcf_latest = float(fcf.sort_index().iloc[-1])
        fcf_yield = fcf_latest / mcap if mcap and not _np.isnan(mcap) and mcap > 0 else _np.nan
    except Exception:
        fcf_latest, fcf_yield = _np.nan, _np.nan

    # EV/EBIT
    try:
        debt_latest = float(total_debt.sort_index().iloc[-1]) if len(total_debt) else _np.nan
        cash_latest = float(cash.sort_index().iloc[-1]) if len(cash) else _np.nan
        ev = (mcap or _np.nan) + (debt_latest or 0) - (cash_latest or 0)
        ebit_latest = float(ebit.sort_index().iloc[-1]) if len(ebit) else _np.nan
        ev_ebit = ev / ebit_latest if (not _np.isnan(ev) and not _np.isnan(ebit_latest) and ebit_latest != 0) else _np.nan
    except Exception:
        ev_ebit = _np.nan

    sector = info.get("sector", "")
    name = info.get("shortName", tk)

    return {
        "ticker": tk,
        "name": name,
        "sector": sector,
        "pe": pe,
        "ev_ebit": ev_ebit,
        "fcf_yield": fcf_yield,
        "sales_cagr": sales_cagr,
        "profit_cagr": profit_cagr,
        "roe": roe,
        "dte": dte,
        "gross_margin": gm_latest,
        "gross_margin_std": gm_std,
        "market_cap": mcap,
        "price": price,
        "ev": ev
    }


# --- Helper: Market snapshot metrics for Investing persistence ---
def _compute_market_snapshot_metrics(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        try:
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(days=420)  # ~ 20 months buffer
            dfp = fetch_prices(tk, start=start, end=end, interval="1d")
            if dfp is None or dfp.empty or 'Close' not in dfp.columns:
                rows.append({"ticker": tk})
                continue
            c = pd.to_numeric(dfp['Close'], errors='coerce').dropna()
            if c.empty:
                rows.append({"ticker": tk})
                continue
            def _ret(n):
                if len(c) <= n:
                    return np.nan
                return float(c.iloc[-1]/c.iloc[-n-1] - 1.0)
            tr3  = _ret(63)   # ~3M
            tr6  = _ret(126)  # ~6M
            tr12 = _ret(252)  # ~12M
            ma200 = c.rolling(200).mean()
            above = float(c.iloc[-1] > ma200.iloc[-1]) if len(ma200.dropna()) else np.nan
            rows.append({
                "ticker": tk,
                "total_return_3m": tr3,
                "total_return_6m": tr6,
                "total_return_12m": tr12,
                "above_ma200": above,
            })
        except Exception:
            rows.append({"ticker": tk})
    return pd.DataFrame(rows)


def render_investing_tab():
    """Render a compact stock screener tab based on the hand-drawn framework."""
    import streamlit as _st
    import pandas as _pd
    import numpy as _np

    _st.header("💼 Investing – Stock Screener (Quality & Price)")

    with _st.expander("🔧 Einstellungen", expanded=True):
        tickers_text = _st.text_area(
            "Tickers (kommagetrennt)",
            value="AAPL, MSFT, NVDA, KO, PG, JNJ, META, GOOGL, ADBE, MA, V"
        )
        # Schwellenwerte – nahe am Poster
        col1, col2, col3 = _st.columns(3)
        with col1:
            min_sales = _st.number_input("Sales CAGR ≥", min_value=0.0, max_value=0.5, value=0.07, step=0.01)
            min_profit = _st.number_input("Profit CAGR ≥", min_value=0.0, max_value=0.5, value=0.07, step=0.01)
        with col2:
            min_roe = _st.number_input("ROE ≥", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
            max_dte = _st.number_input("Debt/Equity ≤", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        with col3:
            max_pe = _st.number_input("P/E ≤", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
            max_ev_ebit = _st.number_input("EV/EBIT ≤", min_value=0.0, max_value=200.0, value=18.0, step=1.0)
            min_fcfy = _st.number_input("FCF‑Yield ≥", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
        # --- Stil-Gewichtung Selectbox ---
        style = _st.selectbox("Stil-Gewichtung", ["garp", "value", "dividend"], index=0)
        _st.session_state.setdefault("invest_style", style)
        _st.session_state["invest_style"] = style

        # 🔰 Bewertung (DCF) – optional aktivieren
        with _st.expander("💰 Bewertung (DCF) – Optionen", expanded=False):
            enable_valuation = _st.checkbox("DCF-Fair-Value im Screener berechnen", value=True)
            colV1, colV2, colV3 = _st.columns(3)
            with colV1:
                dcf_r = _st.number_input("Diskontsatz r", min_value=0.05, max_value=0.18, value=0.10, step=0.01)
            with colV2:
                dcf_years = _st.number_input("Prognosejahre", min_value=3, max_value=7, value=5, step=1)
            with colV3:
                dcf_gterm = _st.number_input("Terminal g∞", min_value=-0.02, max_value=0.03, value=0.02, step=0.005, format="%.3f")

        run = _st.button("🚀 Screen starten")

    # --- Investing UI state ---
    state = _st.session_state
    if 'invest_df' not in state:
        state['invest_df'] = None
    if 'invest_df_scored' not in state:
        state['invest_df_scored'] = None
    if 'invest_selected' not in state:
        state['invest_selected'] = 'AAPL'

    if not run:
        if state['invest_df'] is None:
            _st.info("Konfiguriere die Schwellen & klicke auf **Screen starten** (optional). Unten kannst du auch ohne Screening eine Detailanalyse starten.")
            _st.markdown("---")
            _st.subheader("🔎 Schnell-Analyse (ohne Screening)")
            default_sym = state.get('invest_selected', 'AAPL')
            quick = _st.text_input("Ticker", value=default_sym, key='invest_quick_symbol').strip().upper()
            if quick:
                state['invest_selected'] = quick
                try:
                    render_investing_analysis(quick)
                except Exception as _e:
                    _st.error(f"Analyse fehlgeschlagen: {_e}")
            return
        # Wenn bereits Ergebnisse existieren, verwende Cache und zeige Liste + Detailpanel
        df = state['invest_df'].copy()
        df_scored = state['invest_df_scored'] if state['invest_df_scored'] is not None else df.copy()

    tks = [t.strip().upper() for t in tickers_text.split(',') if t.strip()]
    rows = []
    for tk in tks:
        try:
            rows.append(_compute_screener_metrics(tk))
        except Exception as e:
            rows.append({"ticker": tk, "error": str(e)})

    df = _pd.DataFrame(rows)
    # --- Persist basic master + fundamentals snapshot (best effort) ---
    try:
        comp_cols = ['ticker', 'name', 'sector']
        df_comp = df[comp_cols].dropna(subset=['ticker']).copy()
        df_comp['industry'] = ''
        df_comp['currency'] = ''
        df_comp['country'] = ''
        upsert_companies(df_comp, db_path=DB_PATH)
    except Exception as _e:
        _st.caption(f"(Hinweis) Companies nicht gespeichert: {_e}")

    try:
        as_of_date = _pd.Timestamp.utcnow().date().isoformat()
        fund_cols_map = {
            'revenue_ttm': _np.nan,
            'ebit_ttm': _np.nan,
            'eps_ttm': _np.nan,
            'fcf_ttm': _np.nan,
            'gross_margin': 'gross_margin',
            'op_margin': _np.nan,
            'fcf_margin': _np.nan,
            'roic': _np.nan,
            'roe': 'roe',
            'net_debt': _np.nan,
            'ebitda_ttm': _np.nan,
            'shares_out': _np.nan,
            'dividend_yield': _np.nan,
            'buyback_yield': _np.nan,
            'interest_coverage': _np.nan,
            'capex_ttm': _np.nan,
            'sector': 'sector',
            'currency': ''
        }
        df_fund = _pd.DataFrame({
            'ticker': df['ticker'],
            **{k: (df[v] if isinstance(v, str) and v in df.columns else fund_cols_map[k]) for k, v in
               fund_cols_map.items()}
        })
        upsert_fundamentals_ttm(df_fund, as_of_date=as_of_date, db_path=DB_PATH)
    except Exception as _e:
        _st.caption(f"(Hinweis) Fundamentals_TTM nicht gespeichert: {_e}")

    # --- Enrich with market snapshot metrics (3/6/12M returns, above_ma200) ---
    try:
        snap_df = _compute_market_snapshot_metrics(df['ticker'].dropna().astype(str).tolist())
        if not snap_df.empty:
            df = df.merge(snap_df, on='ticker', how='left')
    except Exception as _e:
        _st.caption(f"(Hinweis) Market Snapshots konnten nicht berechnet werden: {_e}")

    # --- Persist market snapshots (returns & >MA200) and ensure company master is up to date ---
    try:
        if 'ticker' in df.columns:
            # Save market snapshots with current UTC date
            ms_cols = [
                'ticker', 'price', 'market_cap', 'ev',
                'total_return_3m', 'total_return_6m', 'total_return_12m', 'above_ma200'
            ]
            avail_cols = [c for c in ms_cols if c in df.columns]
            ms = df[avail_cols].dropna(subset=['ticker']).copy()
            if not ms.empty:
                ms['snap_date'] = _pd.Timestamp.utcnow().date().isoformat()
                save_market_snapshots(ms)

            # Safety: ensure companies master also contains all screened tickers
            comp_cols = ['ticker', 'name', 'sector']
            comp_df = df[[c for c in comp_cols if c in df.columns]].dropna(subset=['ticker']).copy()
            if not comp_df.empty:
                # fill optional fields if missing
                comp_df['industry'] = comp_df.get('industry', _pd.Series('', index=comp_df.index))
                comp_df['currency'] = comp_df.get('currency', _pd.Series('', index=comp_df.index))
                comp_df['country']  = comp_df.get('country',  _pd.Series('', index=comp_df.index))
                upsert_companies(comp_df, db_path=DB_PATH)
    except Exception as _e:
        _st.caption(f"(Hinweis) Persistierung der Market Snapshots/Companies übersprungen: {_e}")

    # --- DCF Fair Value & Valuation Status (optional) ---
    df["fair_value"] = np.nan
    df["deviation_pct"] = np.nan
    df["valuation_status"] = "n/a"

    if 'enable_valuation' in locals() and enable_valuation:
        _st.caption("💡 DCF wird für jede Aktie berechnet (kann bei vielen Tickern ein paar Sekunden dauern).")
        for tk in df['ticker'].dropna().astype(str).tolist():
            try:
                dcf = estimate_dcf_fair_value(tk, r=float(dcf_r), g_years=int(dcf_years), g_terminal=float(dcf_gterm))
                fv = dcf.get("fair_value", np.nan)
                dev = dcf.get("deviation", None)
                df.loc[df['ticker'] == tk, "fair_value"] = fv
                if dev is not None and not pd.isna(dev):
                    df.loc[df['ticker'] == tk, "deviation_pct"] = float(dev) * 100.0
                    lab, _col = valuation_status_from_deviation(dev)
                    df.loc[df['ticker'] == tk, "valuation_status"] = lab
            except Exception:
                pass

    # Qualitäts- und Preis-Filter anwenden
    quality_mask = (
        (df["sales_cagr"] >= min_sales) &
        (df["profit_cagr"] >= min_profit) &
        (df["roe"] >= min_roe) &
        (df["dte"] <= max_dte)
    )
    moat_hint = (
        (df["gross_margin"] >= 0.25) &
        (df["gross_margin_std"] <= 0.08)
    )
    price_mask = (
        (df["pe"] <= max_pe) &
        (df["ev_ebit"] <= max_ev_ebit) &
        (df["fcf_yield"] >= min_fcfy)
    )

    df["passes_quality"] = quality_mask.fillna(False) & moat_hint.fillna(False)
    df["passes_price"] = price_mask.fillna(False)
    df["final_pick"] = df["passes_quality"] & df["passes_price"]

    _st.subheader("Ergebnisliste")
    show_cols = [
        "ticker","name","sector","market_cap","price",
        # Bewertung
        "fair_value","deviation_pct","valuation_status",
        # Qualität
        "sales_cagr","profit_cagr","roe","dte","gross_margin","gross_margin_std",
        # Preis/Mulitples
        "pe","ev_ebit","fcf_yield",
        # Filterflags
        "passes_quality","passes_price","final_pick"
    ]
    df_show = df[show_cols].copy()

    # --- Scoring & Persistierung ---
    # Baue ein kompaktes DF für das Scoring/Persistieren; fehlende Spalten sind okay (werden als NaN behandelt)
    df_screen = df.copy()
    # Mindestspalten für Persist: ticker, sector, price, market_cap, ev (falls vorhanden), ev_ebit, pe, fcf_yield
    for col in ["ticker","sector","price","market_cap","ev","ev_ebit","pe","fcf_yield",
                "sales_cagr","profit_cagr","total_return_3m","total_return_6m","total_return_12m","above_ma200",
                "fair_value","deviation_pct","valuation_status"]:
        if col not in df_screen.columns:
            df_screen[col] = (np.nan if col != "valuation_status" else "n/a")

    # Scoring nach gewähltem Stil
    chosen_style = _st.session_state.get("invest_style", "garp")
    try:
        df_scored = compute_investing_scores(df_screen, style=chosen_style)
    except Exception as _e:
        df_scored = df_screen.copy()
        _st.warning(f"Scoring konnte nicht vollständig berechnet werden: {_e}")
    # Ergebnisse für spätere Reruns cachen
    state['invest_df'] = df.copy()
    state['invest_df_scored'] = df_scored.copy() if isinstance(df_scored, _pd.DataFrame) else df.copy()

    # Optional: direkt in DB persistieren
    col_save1, col_save2 = _st.columns([1,1])
    do_save = col_save1.checkbox("Ergebnis in DB speichern", value=True)
    show_last = col_save2.checkbox("Letzten Run anzeigen", value=False)

    if do_save:
        try:
            params = {
                "style": chosen_style,
                "filters": {
                    "sales_cagr_min": float(min_sales),
                    "profit_cagr_min": float(min_profit),
                    "roe_min": float(min_roe),
                    "pe_max": float(max_pe),
                    "ev_ebit_max": float(max_ev_ebit),
                    "dte_max": float(max_dte),
                    "fcf_yield_min": float(min_fcfy),
                }
            }
            run_id = persist_investing_run(df_scored, params, style=chosen_style, db_path=DB_PATH)
            _st.caption(f"✅ Screen gespeichert (Run-ID: {run_id}).")
        except Exception as _e:
            _st.warning(f"Persistierung übersprungen: {_e}")

    if show_last:
        try:
            last_id, df_last = load_last_screen_results(db_path=DB_PATH)
            if last_id is not None and not df_last.empty:
                _st.markdown(f"**Letzter gespeicherter Run:** ID {last_id}")
                _st.dataframe(df_last, use_container_width=True)
                _st.download_button(
                    "CSV herunterladen (letzter Run)",
                    data=df_last.to_csv(index=False).encode("utf-8"),
                    file_name=f"screen_results_{last_id}.csv",
                    mime="text/csv"
                )
        except Exception as _e:
            _st.warning(f"Konnte letzten Run nicht laden: {_e}")

    _st.dataframe(df_show)
    _st.caption("Bewertung: Unterbewertet ≤ −20% Abweichung | Fair ±20% | Überbewertet ≥ +20% (gegenüber DCF-Fair-Value).")

    # Downloads
    _st.download_button(
        "📥 CSV herunterladen (alle)",
        data=df_show.to_csv(index=False),
        file_name="stock_screener_results.csv",
        mime="text/csv"
    )

    try:
        _st.download_button(
            "📥 CSV herunterladen (mit Scores)",
            data=(df_scored if 'df_scored' in locals() else df_screen).to_csv(index=False),
            file_name="stock_screener_results_scored.csv",
            mime="text/csv"
        )
    except Exception:
        pass

    _st.success(f"Total: {len(df)} | Final Picks: {int(df['final_pick'].sum())}")

    _st.markdown("---")
    _st.subheader("🔍 Detailanalyse eines Tickers")

    try:
        pick_pool = df.loc[df['ticker'].notna(), 'ticker'].astype(str).unique().tolist()
        final_pool = df.loc[df.get('final_pick', False), 'ticker'].astype(str).unique().tolist()
        default_idx = pick_pool.index(final_pool[0]) if final_pool and final_pool[0] in pick_pool else 0
        prev_sel = state.get('invest_selected')
        if prev_sel in pick_pool:
            idx = pick_pool.index(prev_sel)
        else:
            idx = default_idx
        sel = _st.selectbox("Ticker wählen", options=pick_pool, index=idx, key='invest_selectbox')
        state['invest_selected'] = sel
        render_investing_analysis(sel)
        # --- Bewertung & Peers ----------------------------------------------------
        _st.markdown("---")
        c_top1, c_top2 = _st.columns([2, 1])
        with c_top2:
            try:
                # Verwende die DCF-Parameter aus der UI, mit Defaults als Fallback
                _dcf_r = float(locals().get('dcf_r', 0.10))
                _dcf_years = int(locals().get('dcf_years', 5))
                _dcf_gterm = float(locals().get('dcf_gterm', 0.02))
                dcf_res = estimate_dcf_fair_value(sel, r=_dcf_r, g_years=_dcf_years, g_terminal=_dcf_gterm)
                fv = dcf_res.get('fair_value', float('nan'))
                dev = dcf_res.get('deviation', None)
                label, color = valuation_status_from_deviation(dev)
                _st.markdown(
                    f"""
                    <div style='display:flex; justify-content:flex-end;'>
                      <div style='padding:8px 12px; border-radius:8px; background:{color}; color:white; font-weight:700;'>
                        Bewertung: {label}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as _e:
                _st.caption(f"(Hinweis) DCF-Badge konnte nicht berechnet werden: {_e}")

        _st.subheader("👥 Peer-Group Vergleich (Sektor)")

        try:
            # 1) Sektor des selektierten Tickers bestimmen (erst aus df, sonst via yfinance)
            _sel_sector = None
            try:
                _sel_sector = df.loc[df['ticker'] == sel, 'sector'].dropna().astype(str).iloc[0]
            except Exception:
                pass
            if not _sel_sector:
                try:
                    _t = yf.Ticker(sel)
                    _sel_sector = _t.info.get('sector', '')
                except Exception:
                    _sel_sector = ''

            # 2) Peers aus aktuellem Screener-DF (falls vorhanden)
            peers = pd.DataFrame()
            try:
                peers = df[df['sector'].fillna('') == _sel_sector].copy() if (
                            'df' in locals() and not df.empty) else pd.DataFrame()
            except Exception:
                peers = pd.DataFrame()

            # 3) Falls zu wenige Peers: aus DB 'companies' nachladen (gleicher Sektor)
            if (peers.shape[0] < 2) and _sel_sector:
                try:
                    with get_connection() as _conn:
                        _c = pd.read_sql_query(
                            "SELECT ticker, sector FROM companies WHERE sector = ? ORDER BY ticker LIMIT 50",
                            _conn, params=(_sel_sector,)
                        )
                    have = set(peers['ticker'].astype(str)) if not peers.empty else set()
                    extra_tks = [t for t in _c['ticker'].astype(str).tolist() if t not in have]
                except Exception:
                    extra_tks = []

                # 4) Fehlen Kennzahlen: für bis zu 12 zusätzliche Titel schnell rechnen
                add_rows = []
                for tk in extra_tks[:12]:
                    try:
                        add_rows.append(_compute_screener_metrics(tk))
                    except Exception:
                        continue
                if add_rows:
                    peers = pd.concat([peers, pd.DataFrame(add_rows)],
                                      ignore_index=True) if not peers.empty else pd.DataFrame(add_rows)

            # 5) Sicherstellen, dass selektierter Ticker in der Peer-Tabelle ist
            if peers.empty or (peers['ticker'].astype(str) == sel).sum() == 0:
                try:
                    peers = pd.concat([peers, pd.DataFrame([_compute_screener_metrics(sel)])],
                                      ignore_index=True) if not peers.empty else pd.DataFrame(
                        [_compute_screener_metrics(sel)])
                except Exception:
                    pass

            # 6) Zusätzliche Spalten anreichern: EV/EBITDA, Div.-Rendite, Buyback-Yield
            def _augment_row(row):
                try:
                    t = yf.Ticker(str(row.get('ticker')))
                    info = getattr(t, 'info', {}) or {}

                    # EV/EBITDA
                    ev = row.get('ev', np.nan)
                    ebitda_info = info.get('ebitda', np.nan)
                    row['ev_ebitda'] = (ev / ebitda_info) if (
                                pd.notna(ev) and pd.notna(ebitda_info) and float(ebitda_info) != 0) else np.nan

                    # Dividendenrendite
                    row['div_yield'] = info.get('dividendYield', np.nan)

                    # Buyback-Yield (≈ Repurchase of Stock / MarketCap, positives Vorzeichen = Rückkäufe)
                    try:
                        cf = t.cashflow
                        rep = pd.to_numeric(cf.loc['Repurchase Of Stock'], errors='coerce').dropna().sort_index().iloc[
                            -1]
                        mcap = row.get('market_cap', info.get('marketCap', np.nan))
                        row['buyback_yield'] = (-float(rep) / float(mcap)) if (
                                    pd.notna(rep) and pd.notna(mcap) and float(mcap) > 0) else np.nan
                    except Exception:
                        row['buyback_yield'] = np.nan
                except Exception:
                    row['ev_ebitda'] = row.get('ev_ebitda', np.nan)
                    row['div_yield'] = row.get('div_yield', np.nan)
                    row['buyback_yield'] = row.get('buyback_yield', np.nan)
                return row

            if not peers.empty:
                peers = peers.apply(_augment_row, axis=1)

            # 7) Spaltenauswahl und Sortierung (selektierter Ticker oben)
            base_cols = [
                'ticker', 'name', 'sector', 'market_cap', 'price',
                'pe', 'ev_ebit', 'ev_ebitda', 'fcf_yield', 'div_yield', 'buyback_yield',
                'total_return_3m', 'total_return_6m', 'total_return_12m', 'above_ma200'
            ]
            # DCF/Valuation-Felder mitnehmen, falls im df vorhanden
            for extra in ['fair_value', 'deviation_pct', 'valuation_status']:
                if extra in peers.columns:
                    base_cols.append(extra)

            show_cols = [c for c in base_cols if c in peers.columns]
            peers = peers[show_cols].copy()
            if 'ticker' in peers.columns:
                peers['__sel__'] = peers['ticker'].astype(str).eq(sel)
                peers = peers.sort_values(['__sel__', 'market_cap'], ascending=[False, False]).drop(columns=['__sel__'],
                                                                                                    errors='ignore')

            _st.dataframe(peers, use_container_width=True)

        except Exception as _e:
            _st.caption(f"(Hinweis) Peer-Vergleich nicht möglich: {_e}")

        # --- DB-Viewer: Hilfreiche SQL-Queries ------------------------------------
        with _st.expander("🗄️ DB-Viewer: nützliche SQL-Queries"):
            _st.code(
                f"""
-- Letzte Snapshots für {sel}
SELECT *
FROM market_snapshots
WHERE ticker = '{sel}'
ORDER BY snap_date DESC
LIMIT 30;

-- Companies-Stammdaten für {sel}
SELECT *
FROM companies
WHERE ticker = '{sel}';

-- Letzter Screener-Run (Kopfzeilen)
SELECT *
FROM screen_runs
ORDER BY run_id DESC
LIMIT 1;

-- Ergebnisse des letzten Runs (falls vorhanden)
SELECT *
FROM screen_results
WHERE run_id = (SELECT MAX(run_id) FROM screen_runs)
ORDER BY score DESC;
""",
                language="sql"
            )
    except Exception as _e:
        _st.caption(f"(Hinweis) Analyse konnte nicht gerendert werden: {_e}")

st.set_page_config(layout="wide")
st.title("📊 Marktanalyse-Dashboard: Buy-/Test-Zonen & Sektorrotation")

# View switch: Trading vs Investing
view_mode = st.sidebar.radio("Modus", ["🎯 Trading", "💼 Investing"], index=0)
if view_mode == "💼 Investing":
    render_investing_tab()
    st.stop()  # do not execute the Trading UI below when in Investing mode

ticker = None  # move definition down
st.sidebar.title("🔧 Einstellungen")
interval = st.sidebar.selectbox("⏱️ Datenintervall", options=["1wk", "1d", "1h"], index=0)
# Intervall-Notiz unterhalb des Intervall-Selectbox
resolution_note = {
    "1h": "⏰ Intraday (Scalping/Daytrading)",
    "1d": "🔎 Daily (Swingtrading)",
    "1wk": "📆 Weekly (Makro-Trends)"
}
st.sidebar.markdown(f"**Ausgewähltes Intervall:** {resolution_note.get(interval, '')}")

vereinfachte_trading = st.sidebar.checkbox("🎯 Vereinfachte Trading-Ansicht", value=False)
debug_mode = st.sidebar.checkbox("🔧 Debug anzeigen", value=False)


# Sidebar: Anzeigeoptionen für Indikatoren und Signale
with st.sidebar.expander("🔍 Anzeigen"):
    show_indicators = st.checkbox("Indikatoren anzeigen", value=True, disabled=vereinfachte_trading)
    show_signals = st.checkbox("Buy/Test Signale anzeigen", value=True, disabled=vereinfachte_trading)
    show_fib_extensions = st.checkbox("Fibonacci Extensions anzeigen", value=True, disabled=vereinfachte_trading)

# Neu: Auswahlfeld für Trendrichtung
trend_direction = st.sidebar.radio("Trendrichtung für Fibonacci", options=["Uptrend", "Downtrend"], index=0)

# Dynamische Standardwerte für RSI/MA je nach Intervall
if interval == "1h":
    default_rsi_buy = 35
    default_rsi_test = 70
    default_ma_buy_distance = 2
elif interval == "1wk":
    default_rsi_buy = 45
    default_rsi_test = 60
    default_ma_buy_distance = 5
else:
    default_rsi_buy = 40
    default_rsi_test = 65
    default_ma_buy_distance = 3

ticker = st.sidebar.text_input("📈 Ticker", value="NQ=F")
# Tickerliste expander
with st.sidebar.expander("📘 Tickerliste (Beispiele)"):
    st.markdown("""
    **Indizes**
    - ^GSPC → S&P 500  
    - ^NDX → Nasdaq 100  
    - ^DJI → Dow Jones  
    - ^RUT → Russell 2000  
    - ^GDAXI → Dax 40

    **Einzelaktien**
    - AAPL → Apple  
    - MSFT → Microsoft  
    - NVDA → Nvidia  
    - TSLA → Tesla  
    - AMZN → Amazon
    - AMD → AMD
    - MO.PA → LVMH
    
    **Einzelaktien**
    - GC=F → Gold Future  

    **ETFs**
    - SPY → S&P 500 ETF  
    - QQQ → Nasdaq 100 ETF  
    - IWM → Russell 2000 ETF  
    - DIA → Dow Jones ETF  
    """)


# --- Main Tabs (must be defined before use) ----------------------------------
tab_chart, tab_forecast, tab_log, tab_macro, tab_sector, tab_mtf, tab_live, tab_checklist = st.tabs([
    "📈 Chart & Zonen",
    "🧮 Forecast",
    "🧾 Trading-Log",
    "🌐 Makro",
    "📊 Sektorrotation",
    "🗓️ Multi-TF Candles",
    "⏱️ Live 15m",
    "📝 Disziplin-Checkliste"
])


start_date = st.sidebar.date_input("📅 Startdatum", value=pd.to_datetime("2022-01-01"))
# Set default end date to tomorrow (today + 1 day), but only as default; if the user selects another date, use that.
default_end_date = pd.to_datetime("today") + pd.Timedelta(days=1)
end_date = st.sidebar.date_input("📅 Enddatum", value=default_end_date)

# --- Historical Forecasts (from DB) in Forecast Tab ---
with tab_forecast:
    st.subheader("📜 Historische Forecasts (aus DB)")
    conn_hist = get_connection()
    # Load prices from DB (falls vorhanden)
    prices_query = """
        SELECT date, open, high, low, close, volume
        FROM historical_prices
        WHERE ticker = ?
        ORDER BY date
    """
    prices_df = pd.read_sql(prices_query, conn_hist, params=(ticker,))
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Load forecasts from DB
    forecast_query = """
        SELECT forecast_date, forecast, upper_band, lower_band
        FROM lstm_forecasts
        WHERE ticker = ?
        ORDER BY forecast_date
    """
    forecast_df = pd.read_sql(forecast_query, conn_hist, params=(ticker,))
    conn_hist.close()
    if not prices_df.empty:
        fig_db, ax_db = plt.subplots(figsize=(12, 6))
        ax_db.plot(prices_df['date'], prices_df['close'], label='Close Price', color='black')
        if not forecast_df.empty:
            forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
            ax_db.plot(forecast_df['forecast_date'], forecast_df['forecast'], label='Forecast', linewidth=2)
            ax_db.fill_between(
                forecast_df['forecast_date'],
                forecast_df['lower_band'],
                forecast_df['upper_band'],
                alpha=0.2,
                label='Forecast Band'
            )
        ax_db.set_title(f"{ticker} Preis & Forecast")
        ax_db.set_xlabel("Datum"); ax_db.set_ylabel("Preis")
        ax_db.legend(loc='upper left')
        st.pyplot(fig_db)
        c1, c2 = st.columns(2)
        with c1:
            if st.checkbox("Historische Preisdaten anzeigen", key="show_prices_db"):
                st.dataframe(prices_df)
        with c2:
            if st.checkbox("Forecast-Daten anzeigen", key="show_forecast_db"):
                st.dataframe(forecast_df)
    else:
        st.info("Keine Preisdaten im DB-Cache für diesen Ticker gefunden.")

# --- Trading-Log Tab ---------------------------------------------------------
with tab_log:
    st.subheader("🧾 Trading-Log")

    # Eingabe-Formular (Datum, Produkt, PnL)
    with st.form("trade_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            fdate = st.date_input("Datum", value=pd.to_datetime("today").date())
        with c2:
            fprod = st.text_input("Produkt", value=(ticker or "")).strip()
        with c3:
            fpnl = st.number_input("PnL", value=0.0, step=50.0, format="%.2f")
        submitted = st.form_submit_button("➕ Speichern")

    if submitted:
        try:
            upsert_trading_result(pd.to_datetime(fdate), fprod, float(fpnl))
            st.success("Eintrag gespeichert.")
        except Exception as _e:
            st.error(f"Konnte Eintrag nicht speichern: {_e}")

    # Daten laden & anzeigen
    try:
        tdf = load_trading_results()
    except Exception as _e:
        tdf = pd.DataFrame()
        st.warning(f"Trading-Log konnte nicht geladen werden: {_e}")

    if tdf is not None and not tdf.empty:
        tdf['trade_date'] = pd.to_datetime(tdf['trade_date'])
        tdf = tdf.sort_values('trade_date')
        tdf['cum_pnl'] = tdf['pnl'].cumsum()

        # KPIs
        colk1, colk2, colk3 = st.columns(3)
        total_pnl = float(tdf['pnl'].sum())
        win_rate = float((tdf['pnl'] > 0).mean() * 100.0)
        avg_pnl = float(tdf['pnl'].mean()) if len(tdf) else 0.0
        colk1.metric("Gesamt-PnL", f"{total_pnl:,.0f}")
        colk2.metric("Win-Rate", f"{win_rate:.1f}%")
        colk3.metric("Ø PnL/Trade", f"{avg_pnl:,.0f}")

        # Verlauf (Kumulierter PnL)
        st.line_chart(tdf.set_index('trade_date')['cum_pnl'])
        # Tabelle
        st.dataframe(tdf, use_container_width=True)

        # Download
        st.download_button(
            "📥 CSV herunterladen",
            data=tdf.to_csv(index=False),
            file_name="trading_log.csv",
            mime="text/csv",
        )
    else:
        st.info("Noch keine Einträge vorhanden.")
## Remove sliders for RSI/MA/Volume thresholds, use fixed defaults
rsi_buy_threshold = 30
#rsi_test_threshold = 50
ma_buy_distance = 3
price_bins = 50


zone_prominence = st.sidebar.slider("Prominenz für Zonenfindung", 10, 1000, 300, step=50)
with st.sidebar.expander("ℹ️ Erklärung zur Zonenprominenz"):
    st.markdown("""
    Die **Prominenz** bestimmt, wie **ausgeprägt** ein lokales Hoch oder Tief sein muss, um als Zone erkannt zu werden.

    - **Niedrige Prominenz** (z. B. 100): erkennt viele kleinere Zonen – ideal für **Intraday-Setups**
    - **Hohe Prominenz** (z. B. 600–1000): erkennt nur markante, längerfristige Zonen – geeignet für **Swing- oder Positionstrading**

    **Technischer Hintergrund:** Eine Spitze zählt nur dann als relevant, wenn sie sich um mindestens die gewählte Prominenz **von benachbarten Kurswerten abhebt** (basierend auf `scipy.signal.find_peaks`).
    """)



# === Data Loading & Indicator Helpers ===

@st.cache_data(ttl=600)
def fetch_prices(ticker: str, start, end, interval: str) -> pd.DataFrame:
    """Download raw OHLCV data via yfinance with retries, then sanitize.
    Ensures: DateTimeIndex (UTC naive), strictly increasing, no duplicates, non-negative prices.
    """
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
            if df is None or df.empty:
                raise ValueError("Leere Antwort von yfinance")
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df.dropna(how="any")
            # Validation & cleaning
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()
            numeric_cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
            for c in numeric_cols:
                df = df[df[c].astype(float) >= 0]
            # Remove pathological candles (e.g., >30% gap within bar due to bad ticks)
            if {"High","Low","Close"}.issubset(df.columns):
                span = (df["High"] - df["Low"]).abs() / df["Close"].replace(0, pd.NA)
                df = df[(span < 2.0) | span.isna()]  # drop bars with >200% intrabar span
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    # If we got here, all retries failed
    st.error(f"Fehler beim Laden von {ticker}: {last_err}")
    return pd.DataFrame()

# --- Sequence/time split helpers for LSTM ---
def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return index-based time split boundaries (train/val/test) for a sorted DateTimeIndex df."""
    if df.empty:
        return df.index, df.index, df.index
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.index[:i_train], df.index[i_train:i_val], df.index[i_val:]

def create_sequences_no_crossing(data: np.ndarray, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised sequences X,y *within* one contiguous block (no cross-asset leakage)."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])  # target = next normalized Close
    return np.array(X), np.array(y)

def compute_indicators(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    """Compute technical indicators. If features given, compute only those."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    want = set([f.lower() for f in features]) if features else None
    out = df.copy()
    close = out['Close'].squeeze()

    # MAs
    if want is None or 'ma' in want:
        out['MA50'] = close.rolling(window=50).mean()
        out['MA100'] = close.rolling(window=100).mean()
        out['MA200'] = close.rolling(window=200).mean()
        out['MA20']  = close.rolling(window=20).mean()

    # RSI
    out['Close_Series'] = close
    if want is None or 'rsi' in want:
        out['RSI'] = RSIIndicator(close=out['Close_Series'], window=14).rsi()

    # EMAs
    if want is None or 'ema' in want:
        out['EMA5']  = close.ewm(span=5,  adjust=False).mean()
        out['EMA9']  = close.ewm(span=9,  adjust=False).mean()
        out['EMA14'] = close.ewm(span=14, adjust=False).mean()
        out['EMA50'] = close.ewm(span=50, adjust=False).mean()
        out['EMA69'] = close.ewm(span=69, adjust=False).mean()
        out['EMA_5W'] = close.ewm(span=25,   adjust=False).mean()
        out['EMA_5Y'] = close.ewm(span=1260, adjust=False).mean()

    # Bollinger
    if want is None or 'bb' in want:
        bb = BollingerBands(close=out['Close_Series'], window=20, window_dev=2)
        out['BB_upper'] = bb.bollinger_hband()
        out['BB_lower'] = bb.bollinger_lband()
        out['BB_mid']   = bb.bollinger_mavg()

    # ATR(14) – true average true range
    if want is None or 'atr' in want:
        try:
            atr_ind = AverageTrueRange(high=out['High'], low=out['Low'], close=out['Close_Series'], window=14)
            out['ATR14'] = atr_ind.average_true_range()
        except Exception:
            # Fallback if any column missing
            out['ATR14'] = (out['High'].rolling(14).max() - out['Low'].rolling(14).min())

    return out


# --- Live 15m fetch (for Live tab only) --------------------------------------
@st.cache_data(ttl=900)  # 15 Minuten Cache
def _fetch_15m_df_live(tk: str, days: int = 30) -> pd.DataFrame:
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    try:
        df15 = yf.download(tk, start=start, end=end, interval="15m", auto_adjust=False, progress=False)
        if df15 is None or df15.empty:
            return pd.DataFrame()
        if isinstance(df15.columns, pd.MultiIndex):
            df15.columns = df15.columns.get_level_values(0)
        df15.index = pd.to_datetime(df15.index)
        df15 = df15.dropna().sort_index()
        # Indicators for 15m
        close = df15["Close"].astype(float)
        df15["EMA5"] = close.ewm(span=5, adjust=False).mean()
        df15["EMA9"] = close.ewm(span=9, adjust=False).mean()
        df15["EMA12"] = close.ewm(span=12, adjust=False).mean()
        df15["EMA20"] = close.ewm(span=20, adjust=False).mean()
        from ta.volatility import BollingerBands as _BB_live
        bb = _BB_live(close=close, window=20, window_dev=2)
        df15["BB_UP"], df15["BB_MID"], df15["BB_LO"] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
        from ta.momentum import RSIIndicator as _RSI_live
        df15["RSI15"] = _RSI_live(close=close, window=14).rsi()
        return df15
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)  # cache expires after 10 minutes
def load_data(ticker, start, end, interval):
    raw = fetch_prices(ticker, start, end, interval)
    enriched = compute_indicators(raw)
    return enriched

if st.button("🔄 Daten neu laden"):
    st.cache_data.clear()
data = load_data(ticker, start_date, end_date, interval)
if data is None or data.empty:
    st.warning("Keine Kursdaten geladen. Bitte Ticker/Zeitraum/Intervall prüfen.")
    st.stop()
data.index = pd.to_datetime(data.index)
close_series = data['Close_Series']
# --- DB Insert: historical_prices
upsert_prices(ticker, data)


def identify_zone_ranges(series, prominence=0.5):
    # Buy-Zonen: lokale Tiefs
    lows_idx, _ = find_peaks(-series, prominence=prominence)
    low_levels = sorted(set(round(series[i], -1) for i in lows_idx))  # gerundet für Clustering
    # Test-Zonen: lokale Hochs
    highs_idx, _ = find_peaks(series, prominence=prominence)
    high_levels = sorted(set(round(series[i], -1) for i in highs_idx))  # gerundet für Clustering
    return low_levels, high_levels


raw_buy_levels, raw_test_levels = identify_zone_ranges(close_series, prominence=zone_prominence)
buy_levels = raw_buy_levels
test_levels = raw_test_levels

buy_zone_df = pd.DataFrame({'Level': buy_levels})
test_zone_df = pd.DataFrame({'Level': test_levels})

# Buy-/Test-Zonen (manuell, für Signalpunkte)
buy_zone = data[(close_series < data['MA200'] * (1 + ma_buy_distance / 100)) & (data['RSI'] < rsi_buy_threshold)]
#test_zone = data[(close_series > data['MA50'] * 1.05) & (data['RSI'] > rsi_test_threshold)]

# Fibonacci-Level
low = close_series.min()
high = close_series.max()
fib = {
    "0.0": high,
    "0.236": high - 0.236 * (high - low),
    "0.382": high - 0.382 * (high - low),
    "0.5": high - 0.5 * (high - low),
    "0.618": high - 0.618 * (high - low),
    "0.786": high - 0.786 * (high - low),
    "1.0": low,
}

# Trendrichtung erkennen (robust gegen Pandas-Änderungen in positional indexing)
if close_series.iloc[-1] > close_series.iloc[0]:
    trend = "up"
else:
    trend = "down"

# Trend-Info in der Sidebar anzeigen (nach Definition von trend)
st.markdown(f"**Aktueller Trend:** {'Aufwärts (Uptrend)' if trend == 'up' else 'Abwärts (Downtrend)'}")

# Fibonacci-Extensions berechnen
if trend_direction == "Uptrend":
    fib_ext = {
        "1.236": high + 0.236 * (high - low),
        "1.382": high + 0.382 * (high - low),
        "1.618": high + 0.618 * (high - low),
        "2.0": high + 1.0 * (high - low),
        "2.618": high + 1.618 * (high - low),
    }
else:  # Downtrend
    fib_ext = {
        "1.236": low - 0.236 * (high - low),
        "1.382": low - 0.382 * (high - low),
        "1.618": low - 0.618 * (high - low),
        "2.0": low - 1.0 * (high - low),
        "2.618": low - 1.618 * (high - low),
    }

if trend_direction == "Uptrend":
    fib = {
        "0.0": low,
        "0.236": low + 0.236 * (high - low),
        "0.382": low + 0.382 * (high - low),
        "0.5": low + 0.5 * (high - low),
        "0.618": low + 0.618 * (high - low),
        "0.786": low + 0.786 * (high - low),
        "1.0": high,
    }
else:  # Downtrend
    fib = {
        "0.0": high,
        "0.236": high - 0.236 * (high - low),
        "0.382": high - 0.382 * (high - low),
        "0.5": high - 0.5 * (high - low),
        "0.618": high - 0.618 * (high - low),
        "0.786": high - 0.786 * (high - low),
        "1.0": low,
    }

# --- Confluence Zone Evaluator ---
def evaluate_confluence_zones(data: pd.DataFrame, zones: list, lookahead: int = 10) -> dict:
    """Evaluate whether price touched the opposite side of the band within N bars after first touch.
    Returns hit_rate and avg_forward_return for simple sanity checking.
    """
    if not zones or data.empty:
        return {"count": 0, "hit_rate": np.nan, "avg_forward_return_%": np.nan}
    closes = data['Close_Series']
    hits, rets, n = 0, [], 0
    for z in zones:
        level = z.get('mid', z['level'])
        # find first index where price crosses inside the band
        band_low, band_high = z['low'], z['high']
        touched = data[(closes >= band_low) & (closes <= band_high)]
        if touched.empty:
            continue
        first_ix = touched.index[0]
        # look ahead
        future = closes.loc[first_ix:].iloc[1:1+lookahead]
        if future.empty:
            continue
        n += 1
        # define a naive target: move of +1 ATR from mid in direction of prevailing trend
        if 'ATR14' in data.columns and not data['ATR14'].dropna().empty:
            atr = float(data['ATR14'].dropna().reindex(future.index, method='ffill').iloc[0])
        else:
            atr = float(np.nan)
        # measure simple forward return from first touch to end of window
        ret = (future.iloc[-1] - closes.loc[first_ix]) / closes.loc[first_ix]
        rets.append(float(ret) * 100)
        # hit if price exits band by at least half-ATR
        if not np.isnan(atr):
            if (future.max() >= band_high + 0.5*atr) or (future.min() <= band_low - 0.5*atr):
                hits += 1
        else:
            # fallback: exit band any direction
            if (future.max() > band_high) or (future.min() < band_low):
                hits += 1
    hit_rate = (hits / n) if n else np.nan
    avg_ret = (np.mean(rets) if rets else np.nan)
    return {"count": n, "hit_rate": hit_rate, "avg_forward_return_%": avg_ret}

# --- Auto-Tuning Helper -------------------------------------------------------
def run_auto_tuning(data: pd.DataFrame,
                    fib: dict,
                    ma_series_dict: dict,
                    vol_bins: tuple,
                    prom_list: list[int],
                    score_list: list[int],
                    merge_tol_values: list[float],
                    lookahead: int = 10) -> pd.DataFrame:
    """Grid-Search über Prominenz, Mindestscore, Merge-Tol.; misst Trefferquote und Ø-Rendite."""
    results = []
    if data is None or data.empty:
        return pd.DataFrame()

    for prom in prom_list:
        try:
            zones_raw = find_confluence_zones(
                data, prominence=prom, fibs=fib,
                ma_series_dict=ma_series_dict, vol_bins=vol_bins
            )
        except Exception:
            zones_raw = []
        for mtol in merge_tol_values:
            try:
                zones = merge_overlapping_zones(zones_raw, merge_tol_pct=mtol)
            except Exception:
                zones = zones_raw
            for min_score in score_list:
                zones_f = [z for z in zones if z.get('score', 0) >= min_score]
                ev = evaluate_confluence_zones(data, zones_f, lookahead=lookahead)
                results.append({
                    "prominence": prom,
                    "min_score": int(min_score),
                    "merge_tol_pct": float(mtol),
                    "tested": ev.get("count", 0),
                    "hit_rate%": None if pd.isna(ev.get("hit_rate", np.nan)) else round(100*ev["hit_rate"], 2),
                    "avg_fwd_ret%": None if pd.isna(ev.get("avg_forward_return_%", np.nan)) else round(ev["avg_forward_return_%"], 3)
                })

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Ranking: erst Hit-Rate, dann Ø-Rendite; Mindestanzahl Zonen = 5
    df["tested"] = df["tested"].fillna(0).astype(int)
    df_rank = df[df["tested"] >= 5].copy()
    if df_rank.empty:
        df_rank = df.copy()
    df_rank["_rank_key1"] = df_rank["hit_rate%"].fillna(-1)
    df_rank["_rank_key2"] = df_rank["avg_fwd_ret%"].fillna(-999)
    df_rank = df_rank.sort_values(["_rank_key1", "_rank_key2", "tested"], ascending=[False, False, False])
    df_rank.drop(columns=["_rank_key1","_rank_key2"], inplace=True)
    return df_rank

# --- Neue Confluence Zone Logik ---
def find_confluence_zones(data: pd.DataFrame, prominence=300, fibs=None, ma_series_dict=None, vol_bins: tuple | None = None):
    """
    Identifiziert Confluence Zones mit erweitertem Score & Typ:
    - Score-Punkte (max 5): 1=Reaktion (Prominenz), 2=Fibonacci-Nähe, 3=MA-Nähe, 4=BB-Mid-Nähe, 5=Volumen-Cluster
    - Typ: 'buy' (von Tiefs), 'test' (von Hochs)
    - Bandbreite: max(1.5% vom Level, 0.5 * ATR14) – sofern ATR verfügbar
    """
    if data is None or data.empty:
        return []

    series = data['Close_Series']
    atr = data['ATR14'] if 'ATR14' in data.columns else None
    bb_mid = data['BB_mid'] if 'BB_mid' in data.columns else (data['MA20'] if 'MA20' in data.columns else None)

    # Hilfsfunktionen
    def _is_near_fib(val: float, tol=0.015) -> bool:
        if not fibs:
            return False
        for f in fibs.values():
            try:
                if abs(val - float(f)) / max(1.0, abs(float(f))) < tol:
                    return True
            except Exception:
                continue
        return False

    def _is_near_ma(val: float, idx: int, tol=0.015) -> bool:
        if not ma_series_dict:
            return False
        for _, ma_ser in ma_series_dict.items():
            if ma_ser is None or ma_ser.dropna().empty:
                continue
            if idx < len(ma_ser):
                ma_val = float(ma_ser.iloc[idx])
                if pd.notna(ma_val) and abs(val - ma_val) / max(1.0, abs(ma_val)) < tol:
                    return True
        return False

    def _is_near_bb_mid(val: float, idx: int, tol=0.01) -> bool:
        if bb_mid is None or bb_mid.dropna().empty:
            return False
        if idx < len(bb_mid):
            m = float(bb_mid.iloc[idx])
            return pd.notna(m) and abs(val - m) / max(1.0, abs(m)) < tol
        return False

    def _in_volume_cluster(val: float, top_q: float = 0.75) -> bool:
        if not vol_bins:
            return False
        hist_vals, bin_edges = vol_bins
        if len(hist_vals) == 0:
            return False
        # Finde Bin
        bin_idx = np.digitize([val], bin_edges)[0] - 1
        bin_idx = max(0, min(bin_idx, len(hist_vals) - 1))
        threshold = np.quantile(hist_vals, top_q)
        return hist_vals[bin_idx] >= threshold

    # Reaktionen (Prominenz)
    lows_idx, _ = find_peaks(-series, prominence=prominence)
    highs_idx, _ = find_peaks(series, prominence=prominence)

    zones = []
    for idx_list, ztype in ((lows_idx, 'buy'), (highs_idx, 'test')):
        for i in idx_list:
            lvl = float(series.iloc[i])
            score = 0
            score += 1  # Preisreaktion vorhanden
            if _is_near_fib(lvl):
                score += 1
            if _is_near_ma(lvl, i):
                score += 1
            if _is_near_bb_mid(lvl, i):
                score += 1
            if _in_volume_cluster(lvl):
                score += 1
            # Bandbreite (ATR-basiert, falls vorhanden)
            if atr is not None and not atr.dropna().empty and i < len(atr):
                bw = max(0.015 * lvl, 0.5 * float(atr.iloc[i]))
            else:
                bw = 0.015 * lvl
            zones.append({
                'level': lvl,
                'score': int(score),
                'low': lvl - bw,
                'high': lvl + bw,
                'mid': lvl,
                'type': ztype
            })
    return zones

# --- Merge Overlapping Confluence Zones ---
def merge_overlapping_zones(zones: list, merge_tol_pct: float = 0.01) -> list:
    """Merge overlapping/adjacent zones; aggregate score (max) and widen bounds."""
    if not zones:
        return []
    zones = sorted(zones, key=lambda z: z['mid'])
    merged = []
    cur = zones[0].copy()
    for z in zones[1:]:
        # overlap if ranges intersect or are very close (tolerance percentage of price level)
        tol = merge_tol_pct * max(cur['mid'], z['mid'])
        if z['low'] <= cur['high'] + tol and z['type'] == cur['type']:
            cur['low'] = min(cur['low'], z['low'])
            cur['high'] = max(cur['high'], z['high'])
            cur['mid'] = (cur['low'] + cur['high']) / 2
            cur['score'] = max(cur['score'], z['score'])
        else:
            merged.append(cur)
            cur = z.copy()
    merged.append(cur)
    return merged

# Volumenprofil
hist_vals, bin_edges = np.histogram(close_series, bins=price_bins)
max_volume = max(hist_vals)

# --- Sidebar: Auto-Tuning -----------------------------------------------------
with st.sidebar.expander("🤖 Auto-Tuning (Prominenz & Konfluenz)", expanded=False):
    st.caption("Finde automatisch die Kombination mit hoher Trefferquote und solider Ø-Rendite.")
    prom_values  = st.text_input("Prominenz-Liste", value="100,200,400,600,800")
    score_values = st.text_input("Min-Score-Liste", value="2,3,4,5")
    merge_values = st.text_input("Merge-Toleranz (%)", value="0.5,1.0")
    lookahead_sel = st.number_input("Lookahead (Bars)", min_value=3, max_value=40, value=10, step=1)
    start_tuning = st.button("🔎 Auto-Tuning starten", use_container_width=True)

if 'autotune_df' not in st.session_state:
    st.session_state['autotune_df'] = pd.DataFrame()
    st.session_state['autotune_best'] = None

if start_tuning:
    try:
        prom_list = [int(x.strip()) for x in prom_values.split(',') if x.strip()]
    except Exception:
        prom_list = [100,200,400,600,800]
    try:
        score_list = [int(x.strip()) for x in score_values.split(',') if x.strip()]
    except Exception:
        score_list = [2,3,4,5]
    try:
        merge_tol_values = [float(x.strip())/100.0 for x in merge_values.split(',') if x.strip()]
    except Exception:
        merge_tol_values = [0.005, 0.01]

    ma_series_dict = {'MA200': data.get('MA200'), 'EMA50': data.get('EMA50')}
    vol_bins = (hist_vals, bin_edges)

    with st.spinner("Auto-Tuning läuft…"):
        df_rank = run_auto_tuning(
            data, fib, ma_series_dict, vol_bins,
            prom_list, score_list, merge_tol_values,
            lookahead_sel
        )
    st.session_state['autotune_df'] = df_rank
    st.session_state['autotune_best'] = (df_rank.iloc[0].to_dict() if not df_rank.empty else None)

# --- Interaktiver Chart: Chart & Zonen Tab ---
with tab_chart:
    st.subheader("📈 Interaktiver Chart")
    # Plot: Matplotlib-Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    # Set white backgrounds for figure and axes
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Y-Achsen-Skalierung optimieren: Skalenabstand auf 100 Punkte

    ax.yaxis.set_major_locator(MultipleLocator(100))  # Skalenabstand auf 100 Punkte setzen
    ax.plot(close_series.index, close_series.values, label='Close', linewidth=2.5, color='#00bfff')
    ax.plot(data['MA50'],  label='MA50',  linestyle='-', linewidth=2.0, color='#ffaa00')
    ax.plot(data['MA100'], label='MA100', linestyle='-', linewidth=2.0, color='brown')
    ax.plot(data['MA200'], label='MA200', linestyle='-', linewidth=2.2, color='#ff0000')

    ax.plot(data['EMA5'],  label='EMA5',  linestyle='-', linewidth=1.8, color='#cc00cc')
    ax.plot(data['EMA9'],  label='EMA9',  linestyle='-', linewidth=1.8, color='#b8860b')
    ax.plot(data['EMA14'], label='EMA14', linestyle='-', linewidth=1.8, color='#00cc00')
    ax.plot(data['EMA69'], label='EMA69', linestyle='-', linewidth=1.8, color='#9966ff')
    ax.plot(data['MA20'],  label='MA20',  linestyle='-', linewidth=1.6, color='red')

    ax.plot(data['BB_upper'], label='BB Upper', linestyle='-', linewidth=1.2, color='purple', alpha=0.6)
    ax.plot(data['BB_lower'], label='BB Lower', linestyle='-', linewidth=1.2, color='purple', alpha=0.6)
    ax.plot(data['BB_mid'],   label='BB Mid',   linestyle='-', linewidth=1.0, color='purple', alpha=0.5)

    # Legende gut lesbar – außerhalb rechts
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, facecolor='#111111', edgecolor='#dddddd',
              fontsize=11, ncol=1)

    # Signalpunkte
    ax.scatter(buy_zone.index, close_series.loc[buy_zone.index], label='Buy Zone (Signal)', marker='o', color='green', s=80)
    #ax.scatter(test_zone.index, close_series.loc[test_zone.index], label='Test Zone (Signal)', marker='x', color='red', s=80)

    # Set axis label and tick colors to dark for visibility
    ax.tick_params(axis='x', colors='#111111')
    ax.tick_params(axis='y', colors='#111111')
    ax.xaxis.label.set_color('#111111')
    ax.yaxis.label.set_color('#111111')
    ax.title.set_color('#111111')


    ## Remove all sliders for RSI, MA-Nähe, Y-Achsen-Zoom, Clustering-Schwelle und Volume-Bins
    # (No explicit code for these; ensure only zone_prominence and min_score slider remain)

    # Replace previous slider for minimal zone score threshold with new "Konfluenz-Schwelle"
    selected_min_score = st.sidebar.slider("Konfluenz-Schwelle", 1, 3, 2)



    # Sidebar: Erklärung Confluence Zone
    with st.sidebar.expander("ℹ️ Erklärung: Confluence Zone"):
        st.markdown("""
        Eine **Confluence Zone** entsteht, wenn mehrere technische Indikatoren (z. B. Fibonacci, gleitende Durchschnitte, Volumencluster) im selben Preisbereich zusammenfallen. 
        Diese Zonen gelten als besonders relevant für mögliche Umkehrpunkte oder Breakouts.
        """)


    with st.sidebar.expander("🤖 Automatisches Multimarkt-LSTM-Training"):
        st.markdown("""
        Trainiere dein LSTM-Modell **automatisch** mit mehreren Märkten (z. B. S&P 500, Nasdaq, Dow, Russell, AAPL, MSFT, NVDA, TSLA).

        Dadurch wird das Modell robuster und erkennt Muster über verschiedene Indizes und große Aktien hinweg.
        """)

        if st.button("🔄 Modell mit mehreren Märkten trainieren"):
            st.info("📥 Lade kombinierte Daten (mehrere Indizes & Aktien)...")

            base_tickers = ["^GSPC", "^NDX", "^DJI", "^RUT", "AAPL", "MSFT", "NVDA", "TSLA"]
            frames = {}

            for ticker_symbol in base_tickers:
                df = fetch_prices(ticker_symbol, start=start_date, end=end_date, interval="1d")
                if df.empty or 'Close' not in df.columns:
                    continue
                # Feature engineering per-asset
                tmp = pd.DataFrame(index=df.index)
                tmp['Close_Series'] = df['Close'].astype(float)
                tmp['EMA5'] = tmp['Close_Series'].ewm(span=5, adjust=False).mean()
                tmp['MA20'] = tmp['Close_Series'].rolling(window=20).mean()
                tmp['MA50'] = tmp['Close_Series'].rolling(window=50).mean()
                tmp['RSI'] = RSIIndicator(close=tmp['Close_Series'], window=14).rsi()
                # Join VIX features
                vix_df = fetch_prices("^VIX", start=start_date, end=end_date, interval="1d")
                if not vix_df.empty:
                    vix_df = vix_df.rename(columns={"Close": "VIX_Close"})
                    vix_df['VIX_SMA5'] = vix_df['VIX_Close'].rolling(5).mean()
                    vix_df['VIX_RSI'] = RSIIndicator(close=vix_df['VIX_Close'].squeeze(), window=14).rsi()
                    vix_df['VIX_Change'] = vix_df['VIX_Close'].pct_change()
                    vix_df['Month'] = vix_df.index.month / 12.0
                    vix_df = vix_df[['VIX_Close','VIX_SMA5','VIX_RSI','VIX_Change','Month']]
                    tmp = tmp.join(vix_df, how='left')
                # Extra deltas
                tmp['RSI_Change'] = tmp['RSI'].diff()
                tmp['Close_MA20_Pct'] = (tmp['Close_Series'] - tmp['MA20']) / tmp['MA20']
                tmp['Close_EMA5_Pct'] = (tmp['Close_Series'] - tmp['EMA5']) / tmp['EMA5']
                tmp.dropna(inplace=True)
                frames[ticker_symbol] = tmp

            if not frames:
                st.error("❌ Keine gültigen Daten gefunden.")
            else:
                # Fit scaler on the concatenated feature space but build sequences per asset to avoid cross-asset leakage
                feature_cols = ['Close_Series','RSI','MA50','VIX_Close','VIX_SMA5','VIX_RSI','VIX_Change','Month','RSI_Change','Close_MA20_Pct','Close_EMA5_Pct']
                concat_for_scaler = pd.concat([f[feature_cols] for f in frames.values()], axis=0).dropna()
                scaler = MinMaxScaler()
                scaler.fit(concat_for_scaler.values)
                joblib.dump(scaler, 'lstm_scaler.pkl')

                # Build sequences per asset, then stack
                X_list, y_list = [], []
                for tk, fdf in frames.items():
                    feats = fdf[feature_cols].dropna().values
                    feats_scaled = scaler.transform(feats)
                    X_tk, y_tk = create_sequences_no_crossing(feats_scaled, seq_len=30)
                    if len(X_tk) > 0:
                        X_list.append(X_tk)
                        y_list.append(y_tk)
                if not X_list:
                    st.error("❌ Zu wenige Sequenzen für das Training.")
                else:
                    X_all = np.concatenate(X_list, axis=0)
                    y_all = np.concatenate(y_list, axis=0)

                    # Time-based split on concatenated sequences (approximate, since each block is contiguous)
                    n = len(X_all)
                    i_train = int(n * 0.75)
                    i_val = int(n * 0.9)
                    X_train, y_train = X_all[:i_train], y_all[:i_train]
                    X_val,   y_val   = X_all[i_train:i_val], y_all[i_train:i_val]
                    X_test,  y_test  = X_all[i_val:], y_all[i_val:]

                    expected_shape = (X_train.shape[1], X_train.shape[2])
                    model = Sequential()
                    model.add(LSTM(96, activation='tanh', input_shape=expected_shape, return_sequences=False))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')

                    checkpoint = ModelCheckpoint("lstm_model.keras", monitor='val_loss', save_best_only=True, verbose=0)
                    import tensorflow as tf
                    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

                    epochs = 60
                    batch_size = 64
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for epoch in range(epochs):
                        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=batch_size, verbose=0, callbacks=[checkpoint, early_stop])
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Training... Epoche {epoch + 1}/{epochs} | val_loss={hist.history['val_loss'][0]:.6f}")
                        # Break early if early_stop triggered (monitored via patience)
                        if len(model.history.history.get('val_loss', [])) > 0 and early_stop.stopped_epoch:
                            break

                    # Simple out-of-sample metric
                    test_mse = float(model.evaluate(X_test, y_test, verbose=0)) if len(X_test) else float('nan')
                    st.success(f"✅ Multimarkt-Training abgeschlossen. Test-MSE: {test_mse:.6f}")

    # Buy-/Test-Zonen als Flächen (je 1 Rechteck pro Zone mit 1.5% Bandbreite)
    valid_ma200 = data['MA200'].dropna()
    if not valid_ma200.empty:
        buy_center = valid_ma200.mean()
        buy_lower = buy_center * (1 - 0.015)
        buy_upper = buy_center * (1 + 0.015)
        ax.axhspan(buy_lower, buy_upper, color='#00ff00', alpha=0.1, label='Buy-Zone (MA200±1.5%) [manuell]')

    valid_ma50 = data['MA50'].dropna()
    if not valid_ma50.empty:
        test_center = valid_ma50.mean()
        test_lower = test_center * 1.03
        test_upper = test_center * 1.08
        # Adjust test zone to 1.5% band around mean of test zone range for consistency
        test_mid = (test_lower + test_upper) / 2
        test_lower = test_mid * (1 - 0.015)
        test_upper = test_mid * (1 + 0.015)
        ax.axhspan(test_lower, test_upper, color='#ff6600', alpha=0.1, label='Test-Zone (MA50+1.5%) [manuell]')

    # Automatisch erkannte Buy-/Test-Zonen als Rechtecke (je 1 Rechteck pro Zone mit 1.5% Bandbreite)
    if buy_levels:
        buy_min = min(buy_levels)
        buy_max = max(buy_levels)
        buy_lower_auto = buy_min * (1 - 0.015)
        buy_upper_auto = buy_max * (1 + 0.015)
        ax.axhspan(buy_lower_auto, buy_upper_auto, color='#00ff00', alpha=0.1, label='Buy-Zone automatisch')

    if test_levels:
        test_min = min(test_levels)
        test_max = max(test_levels)
        test_lower_auto = test_min * (1 - 0.015)
        test_upper_auto = test_max * (1 + 0.015)
        ax.axhspan(test_lower_auto, test_upper_auto, color='#ff6600', alpha=0.1, label='Test-Zone automatisch')

    # Bestimme Confluence Zones mit Score und merge sie zu breiten Bändern
    ma_series_dict = {'MA200': data['MA200'], 'EMA50': data['EMA50']}
    confluence_zones = find_confluence_zones(
        data, prominence=zone_prominence, fibs=fib, ma_series_dict=ma_series_dict, vol_bins=(hist_vals, bin_edges)
    )
    # Merge overlapping zones to form broader bands
    confluence_zones = merge_overlapping_zones(confluence_zones, merge_tol_pct=0.01)
    # --- DB Insert: confluence_zones
    upsert_zones(ticker, confluence_zones)
    # Zeichne Confluence Zones als horizontale Linien mit neuem Label-Stil
    #
    # --- Neue Beschriftung der Confluence Zones weiter rechts, mit Preisbereich, automatischer Versatz ---
    used_y_positions = []
    min_vsep = 0.01  # minimaler vertikaler Abstand (relativ zum Preis)

    for i, zone in enumerate(confluence_zones):
        color = {3: 'darkgreen', 2: 'orange', 1: 'gray'}.get(zone['score'], 'gray')
        ax.axhline(y=zone['level'], color=color, linestyle='--', linewidth=2, alpha=0.8)
        # Preisbereich (Mitte, Low, High)
        zone_bottom = zone.get('low', zone['level'])
        zone_top = zone.get('high', zone['level'])
        price_level = (zone_top + zone_bottom) / 2
        match_count = zone['score']
        total_indicators = 3
        # Calculate price_min and price_max for annotation
        price_min = min(zone_top, zone_bottom)
        price_max = max(zone_top, zone_bottom)
        # X-Position: deutlich weiter rechts, um Überlappung mit Candles zu vermeiden
        x_pos = data.index[-1] + pd.Timedelta(days=30)
        # Automatischer Versatz bei Überlappung
        y_pos = price_level
        for prev_y in used_y_positions:
            if abs(prev_y - y_pos) / max(1, y_pos) < min_vsep:
                y_pos += (zone_top - zone_bottom) * 0.3 if (i % 2 == 0) else -(zone_top - zone_bottom) * 0.3
        used_y_positions.append(y_pos)
        # Updated label: show score and price range (rounded, upper–lower)
        label = f"Confluence Zone: {match_count}/{total_indicators}\n{zone['low']:.0f}–{zone['high']:.0f}"
        ax.annotate(
            label,
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos + (zone_top-zone_bottom)*0.1),
            ha="left",
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=12,
            arrowprops=None
        )
        # --- Kursziel unterhalb der aktuellen Zone anzeigen ---
        # Verwende echten ATR(14) aus den Indikatoren
        atr_value = float(data['ATR14'].dropna().iloc[-1]) if 'ATR14' in data.columns and not data['ATR14'].dropna().empty else float('nan')
        # Kursziel: Unterkante der Zone - ATR * 1.5
        kursziel = zone_bottom - (atr_value * 1.5) if atr_value == atr_value else zone_bottom  # NaN-safe
        # Kursziel anzeigen (z. B. als Text rechts im Chart)
        ax.text(
            data.index[-1] + pd.Timedelta(days=20),  # Position rechts neben letztem Kerzenstand
            kursziel,
            f"Zielbereich: {kursziel:.0f}" if isinstance(kursziel, (int, float)) else f"Zielbereich: {float(kursziel.iloc[-1]):.0f}",
            verticalalignment='center',
            bbox=dict(facecolor='gray', edgecolor='#111111', boxstyle='round,pad=0.4'),
            fontsize=12,
            color='white'
        )

    custom_lines = [
        Line2D([0], [0], color='darkgreen', lw=2, linestyle='--', label='Confluence Zone (3/3)'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Confluence Zone (2/3)'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Confluence Zone (1/3)'),
    ]

    # Fibonacci farbig in grau (#cccccc), Label oben links, kleinere Schrift
    for lvl, val in fib.items():
        ax.axhline(val, linestyle='--', alpha=0.7, label=f'Fib {lvl} → {val:.0f}', color='#cccccc')
    for lvl, val in fib.items():
        ax.text(data.index.min(), val, f'Fib {lvl}', color='#666666', fontsize=10, verticalalignment='bottom', horizontalalignment='left')

    # Volumenprofil
    for count, edge in zip(hist_vals, bin_edges[:-1]):
        ax.barh(y=edge, width=(count / max_volume) * close_series.max() * 0.1, height=(bin_edges[1] - bin_edges[0]), alpha=0.2, color='gray')

# --- Zusätzliche Makro-Charts ---

# JNK vs SPX Chart mit RSI
def plot_jnk_spx_chart():
    import matplotlib.pyplot as plt
    from ta.momentum import RSIIndicator
    import streamlit as st

    # Daten abrufen
    jnk = fetch_prices("JNK", start=pd.to_datetime("2023-06-01"), end=pd.to_datetime("today"), interval="1d")
    spx = fetch_prices("^GSPC", start=pd.to_datetime("2023-06-01"), end=pd.to_datetime("today"), interval="1d")

    # RSI berechnen
    jnk['RSI'] = RSIIndicator(close=jnk['Close'].squeeze(), window=14).rsi()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2.5, 1]}, sharex=True)
    # Set white backgrounds and dark text/ticks
    fig.patch.set_facecolor("white")
    for ax in axs:
        ax.set_facecolor("white")
        ax.tick_params(axis='x', colors='#111111')
        ax.tick_params(axis='y', colors='#111111')
        ax.xaxis.label.set_color('#111111')
        ax.yaxis.label.set_color('#111111')
        ax.title.set_color('#111111')

    # RSI
    axs[0].plot(jnk.index, jnk['RSI'], color='red', label='RSI (14)')
    axs[0].axhline(70, color='gray', linestyle='--', linewidth=1)
    axs[0].axhline(30, color='gray', linestyle='--', linewidth=1)
    axs[0].set_ylabel('RSI')
    axs[0].legend(loc='upper left')

    # Candlestick (vereinfacht als Linienchart)
    axs[1].plot(jnk.index, jnk['Close'], color='green', label='JNK Close')
    axs[1].set_ylabel('JNK')
    axs[1].legend(loc='upper left')

    # SPX
    axs[2].plot(spx.index, spx['Close'], color='cyan', label='SPX Close')
    axs[2].set_ylabel('SPX')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)

def plot_hyg_chart():
    from ta.momentum import RSIIndicator
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Zeitraum festlegen
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=2)

    # Daten laden
    hyg = fetch_prices("HYG", start=start, end=end, interval="1d")
    spx = fetch_prices("^GSPC", start=start, end=end, interval="1d")

    # RSI für HYG berechnen
    rsi = RSIIndicator(close=hyg["Close"].squeeze()).rsi()
    hyg["RSI"] = rsi

    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # Set white backgrounds and dark text/ticks
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    ax1.title.set_color('black')
    ax2.title.set_color('black')

    # Preisplot: HYG (linke Achse), SPX (rechte Achse)
    ax1.plot(hyg.index, hyg["Close"], label="HYG", color="green")
    ax1.set_ylabel("HYG", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    ax1b = ax1.twinx()
    ax1b.plot(spx.index, spx["Close"], label="SPX", color="blue", alpha=0.6)
    ax1b.set_ylabel("SPX", color="blue")
    ax1b.tick_params(axis="y", labelcolor="blue")

    ax1.set_title("HYG vs SPX (2 Jahre)")
    ax1.grid(True)

    # RSI-Plot
    ax2.plot(hyg.index, hyg["RSI"], label="RSI (HYG)", color="red")
    ax2.axhline(70, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(30, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("RSI")
    ax2.set_title("HYG RSI")
    ax2.grid(True)

    fig.tight_layout()
    st.pyplot(fig)

def plot_vix_chart():
    import matplotlib.pyplot as plt
    from ta.momentum import RSIIndicator

    # Zeitraum: 2 Jahre
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=2)

    vix = fetch_prices("^VIX", start=start, end=end, interval="1d")
    if vix.empty or 'Close' not in vix.columns:
        st.warning("VIX-Daten konnten nicht geladen werden.")
        return

    vix['RSI'] = RSIIndicator(close=vix['Close'].squeeze(), window=14).rsi()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # Set white backgrounds and dark text/ticks
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    ax1.title.set_color('black')
    ax2.title.set_color('black')

    # VIX Close
    ax1.plot(vix.index, vix['Close'], label='VIX Close', color='orange')
    ax1.set_title('VIX (2 Jahre)')
    ax1.set_ylabel('Index Level')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # RSI
    ax2.plot(vix.index, vix['RSI'], label='RSI (14)', color='red')
    ax2.axhline(70, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(30, color='gray', linestyle='--', linewidth=1)
    ax2.set_ylabel('RSI')
    ax2.set_title('VIX RSI')
    ax2.grid(True)

    fig.tight_layout()
    st.pyplot(fig)


def plot_vix_spx_comparison():
    import matplotlib.pyplot as plt
    from ta.momentum import RSIIndicator

    # Zeitraum: 2 Jahre
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=2)

    vix = fetch_prices("^VIX", start=start, end=end, interval="1d")
    spx = fetch_prices("^GSPC", start=start, end=end, interval="1d")

    if vix.empty or spx.empty:
        st.warning("VIX/SPX Daten konnten nicht geladen werden.")
        return

    # Normiere SPX für vergleichbare Skala (0..1)
    spx_norm = (spx['Close'] - spx['Close'].min()) / (spx['Close'].max() - spx['Close'].min())
    # Optional: RSI von VIX als Stimmungsindikator
    vix_rsi = RSIIndicator(close=vix['Close'].squeeze(), window=14).rsi()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    # Set white backgrounds and dark text/ticks
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax1.xaxis.label.set_color('black')
    ax1.yaxis.label.set_color('black')
    ax2.xaxis.label.set_color('black')
    ax2.yaxis.label.set_color('black')
    ax1.title.set_color('black')
    ax2.title.set_color('black')

    # VIX und normierter SPX (sekundäre Achse vermeiden: gleiches 0..1 Maßstab für SPX)
    ax1.plot(vix.index, vix['Close'], label='VIX Close', color='orange')
    ax1b = ax1.twinx()
    ax1b.plot(spx.index, spx_norm, label='SPX (normiert 0–1)', color='blue', alpha=0.6)
    ax1.set_title('VIX vs. S&P 500 (2 Jahre)')
    ax1.set_ylabel('VIX')
    ax1b.set_ylabel('SPX (normiert)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')

    # VIX RSI unten
    ax2.plot(vix.index, vix_rsi, label='VIX RSI (14)', color='red')
    ax2.axhline(70, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(30, color='gray', linestyle='--', linewidth=1)
    ax2.set_ylabel('RSI')
    ax2.set_title('VIX RSI')
    ax2.grid(True)

    fig.tight_layout()
    st.pyplot(fig)


## Removed plot_vix_seasonality and replaced with static image display in Macro tab below


with tab_macro:
    st.subheader("🌐 Makro – Zusatzcharts")

    st.markdown("### JNK vs SPX (inkl. RSI)")
    plot_jnk_spx_chart()

    st.markdown("### HYG vs SPX (2 Jahre)")
    plot_hyg_chart()

    st.markdown("### VIX Chart")
    plot_vix_chart()

    st.markdown("### VIX vs S&P 500")
    plot_vix_spx_comparison()

    # Display S&P 500 and VIX seasonality images in Macro tab
    st.subheader("📊 S&P 500 Seasonality")
    sp500_image_path = os.path.join("images", "S&P500Seasonality.png")
    if os.path.exists(sp500_image_path):
        st.image(sp500_image_path, use_container_width=True)
    else:
        st.warning("S&P 500 Seasonality image not found.")

    st.subheader("📊 VIX Seasonality")
    vix_image_path = os.path.join("images", "vix_seasonality.png")
    if os.path.exists(vix_image_path):
        st.image(vix_image_path, use_container_width=True)
    else:
        st.warning("VIX Seasonality image not found.")



# --- Sector Rotation (configurable) -------------------------------------------
def render_sector_rotation_panel(region: str = "US", custom_list: str | None = None):
    st.subheader(f"📊 Sektorrotation – {region}")
    st.caption("Kennzahlen: 1W/1M/3M/6M Total Returns und 1M Outperformance vs. Benchmark.")

    # Presets
    if region == "US":
        sectors = [
            ("XLK","Tech"), ("XLY","Cons. Discr."), ("XLC","Comm."), ("XLF","Financials"),
            ("XLV","Health"), ("XLI","Industrials"), ("XLP","Cons. Staples"), ("XLE","Energy"),
            ("XLU","Utilities"), ("XLRE","Real Estate"), ("XLB","Materials")
        ]
        bench = ("SPY","S&P 500")
    else:
        # Europe preset: provide placeholders; user can override via custom_list
        sectors = [
            ("EXSA.DE","Europe 600"),  # broad; kept for reference if custom list empty
        ]
        bench = ("EXSA.DE","STOXX Europe 600 ETF")

    # Allow overriding via custom list
    if custom_list:
        pairs = []
        for token in custom_list.split(','):
            t = token.strip()
            if not t:
                continue
            # Accept forms: "TICKER" or "TICKER|Label"
            if '|' in t:
                tk, lbl = t.split('|', 1)
                pairs.append((tk.strip(), lbl.strip()))
            else:
                pairs.append((t, t))
        if pairs:
            sectors = pairs

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(days=420)

    spy = fetch_prices(bench[0], start=start, end=end, interval="1d")
    if spy.empty or 'Close' not in spy.columns:
        st.warning(f"Benchmark {bench[0]} konnte nicht geladen werden.")
        return
    spy_close = pd.to_numeric(spy['Close'], errors='coerce').dropna()

    rows = []
    periods = {"1W":5, "1M":21, "3M":63, "6M":126}

    for tk, name in sectors:
        df = fetch_prices(tk, start=start, end=end, interval="1d")
        if df.empty or 'Close' not in df.columns:
            rows.append({"Ticker": tk, "Sektor": name, **{p: np.nan for p in periods}, "Outperf 1M": np.nan})
            continue
        c = pd.to_numeric(df['Close'], errors='coerce').dropna()
        rec = {"Ticker": tk, "Sektor": name}
        for lab, n in periods.items():
            rec[lab] = float(c.iloc[-1]/c.iloc[-n-1] - 1.0) if len(c) > n else np.nan
        if len(c) > 21 and len(spy_close) > 21:
            sec_1m = float(c.iloc[-1]/c.iloc[-22] - 1.0)
            spy_1m = float(spy_close.iloc[-1]/spy_close.iloc[-22] - 1.0)
            rec["Outperf 1M"] = sec_1m - spy_1m
        else:
            rec["Outperf 1M"] = np.nan
        rows.append(rec)

    df_rot = pd.DataFrame(rows)
    if df_rot.empty:
        st.info("Keine Sektor-Daten.")
        return

    # Heatmap
    hm_cols = ["1W","1M","3M","6M","Outperf 1M"]
    df_hm = df_rot.set_index(['Sektor','Ticker'])[hm_cols]
    fig_hm = go.Figure(data=go.Heatmap(
        z=df_hm.to_numpy(dtype=float),
        x=hm_cols,
        y=[f"{i[0]} ({i[1]})" for i in df_hm.index],
        colorscale='RdYlGn', zmid=0, colorbar=dict(title="Return")
    ))
    fig_hm.update_layout(height=520, margin=dict(l=60,r=20,t=40,b=40), template='plotly_white')
    st.plotly_chart(fig_hm, use_container_width=True)

    # Scatter: Outperf vs. 1M Return
    try:
        x = df_rot["Outperf 1M"].astype(float)
        y = df_rot["1M"].astype(float)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=x, y=y, mode='markers+text', text=df_rot['Ticker'], textposition='top center',
            marker=dict(size=12, line=dict(width=1, color='#333')), hovertext=df_rot['Sektor']
        ))
        fig_sc.add_vline(x=0, line=dict(color='#888', dash='dot'))
        fig_sc.add_hline(y=0, line=dict(color='#888', dash='dot'))
        fig_sc.update_layout(template='plotly_white', height=440,
                             xaxis_title='Outperformance ggü. Benchmark (1M)', yaxis_title='Total Return (1M)')
        st.plotly_chart(fig_sc, use_container_width=True)
    except Exception:
        pass

    st.dataframe(
        df_rot.sort_values("1M", ascending=False)
              .style.format({c: "{:+.2%}" for c in ["1W","1M","3M","6M","Outperf 1M"]}),
        use_container_width=True
    )



# --- Sector Rotation Tab Content ---
with tab_sector:
    st.subheader("📊 Sektorrotation")
    region = st.selectbox("Region", ["US", "Europe"], index=0)
    st.caption("Optional: eigene Liste überschreibt Preset. Format: 'TICKER|Label, TICKER|Label, ...' oder nur 'TICKER' ")
    custom = st.text_input("Eigene Sektor-/ETF-Liste (optional)", value="" if region=="US" else "EXSA.DE|Europe 600")
    render_sector_rotation_panel(region=region, custom_list=custom)





from ta.volatility import BollingerBands as _BB

def _prep_tf_df(symbol: str, interval: str, years: int = 12):
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False)
    if df.empty:
        return df
    # Ensure simple columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    close = df['Close'].squeeze()
    # EMA(5) on this timeframe
    df['EMA5'] = close.ewm(span=5, adjust=False).mean()
    # Bollinger Bands 20, 2
    bb = _BB(close=close, window=20, window_dev=2)
    df['BB_MID'] = bb.bollinger_mavg()
    df['BB_UP'] = bb.bollinger_hband()
    df['BB_LO'] = bb.bollinger_lband()
    # Extras: RSI(14) und längere MA für Kontext
    try:
        from ta.momentum import RSIIndicator as _RSI
        df['RSI14'] = _RSI(close=close, window=14).rsi()
    except Exception:
        df['RSI14'] = np.nan
    # MA200 (auf diesem TF; bei Monthly/Quarterly ist das eher eine sehr lange Glättung)
    try:
        df['MA200'] = close.rolling(window=200).mean()
    except Exception:
        pass
    return df

def _apply_plotly_theme(fig, title: str = None, height: int = 560, showlegend: bool = True):
    fig.update_layout(
        template='plotly_white',
        title=title if title else fig.layout.title.text if fig.layout.title else None,
        height=height,
        margin=dict(l=70, r=200, t=60, b=60),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#111111", size=12),
        legend=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#cfcfcf', borderwidth=1,
            orientation='h', x=0, y=1.05, font=dict(size=11, color='#111111')
        ),
        hovermode="x unified",
        showlegend=showlegend,
        dragmode='zoom',
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#e0e0e0',
        showline=True, linewidth=1, linecolor='#888',
        rangeslider_visible=False,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showspikes=True, spikemode='across', spikecolor='#777', spikethickness=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#eaeaea',
        showline=True, linewidth=1, linecolor='#888',
        side='right',
        showspikes=True, spikemode='across', spikecolor='#777', spikethickness=1,
        zeroline=False,
    )
    return fig

from plotly.subplots import make_subplots as _make_subplots_tf

def _make_tf_figure(df: pd.DataFrame, title: str, show_rsi: bool = True) -> go.Figure:
    # Subplots: Preis + (optional) RSI
    if show_rsi:
        fig = _make_subplots_tf(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.76, 0.24])
    else:
        fig = _make_subplots_tf(rows=1, cols=1)

    # Candles – kräftig, kontrastreich
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='rgb(0,155,0)', decreasing_line_color='rgb(200,0,0)',
        increasing_line_width=2.2, decreasing_line_width=2.2,
        name='Candles'
    ), row=1, col=1)

    # EMA(5)
    if 'EMA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], name='EMA(5)', mode='lines', line=dict(width=2.2, color='rgb(90,0,180)')), row=1, col=1)
    # MA200 (Kontext)
    if 'MA200' in df.columns and df['MA200'].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA(200)', mode='lines', line=dict(width=2.0, color='rgb(220,120,0)', dash='solid')), row=1, col=1)

    # Bollinger: Linien + dezente blaue Fläche
    if {'BB_UP','BB_MID','BB_LO'}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], name='BB Upper', mode='lines', line=dict(dash='dash', width=1.4, color='#2a5bd7')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MID'], name='BB Mid',   mode='lines', line=dict(dash='dot',  width=1.0, color='#2a5bd7'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LO'], name='BB Lower', mode='lines', line=dict(dash='dash', width=1.4, color='#2a5bd7'), fill='tonexty', fillcolor='rgba(42,91,215,0.12)'), row=1, col=1)

    # Last‑price Linie + Badge rechts
    try:
        y_last = float(pd.to_numeric(df['Close'], errors='coerce').dropna().iloc[-1])
        x_last = pd.to_datetime(df.index.max())
        x_label = x_last + pd.Timedelta(days=30)
        fig.add_hline(y=y_last, line=dict(color='#222', width=1, dash='dot'), row=1, col=1)
        fig.add_shape(type='line', x0=x_last, x1=x_label, y0=y_last, y1=y_last, line=dict(color='#222', width=1, dash='dot'), row=1, col=1)
        fig.add_annotation(x=x_label, y=y_last, text=f"Close: {y_last:.0f}", showarrow=False,
                           font=dict(color='#111', size=12), bgcolor='rgba(255,255,255,0.96)',
                           bordercolor='#222', borderwidth=1, xanchor='left', yanchor='middle', row=1, col=1)
    except Exception:
        pass

    # RSI Subplot
    if show_rsi and 'RSI14' in df.columns:
        rsi = pd.to_numeric(df['RSI14'], errors='coerce')
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI(14)', mode='lines', line=dict(width=1.6, color='rgb(180,0,0)')), row=2, col=1)
        # 70/30 Linien
        fig.add_hline(y=70, line=dict(color='#aaaaaa', width=1, dash='dot'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='#aaaaaa', width=1, dash='dot'), row=2, col=1)

    # Theme & Achsen
    _apply_plotly_theme(fig, title=title, height=620 if show_rsi else 520, showlegend=True)
    return fig

def render_multi_tf_candles(symbol: str):
    # Control: RSI-Panel anzeigen
    show_rsi_tf = st.checkbox("RSI‑Panel anzeigen", value=True, help="RSI(14) als Unterchart pro Zeitebene einblenden")
    # Daily, Weekly, Monthly, Quarterly (approx via 3mo)
    d = _prep_tf_df(symbol, '1d', years=3)
    w = _prep_tf_df(symbol, '1wk', years=8)
    m = _prep_tf_df(symbol, '1mo', years=12)
    q = _prep_tf_df(symbol, '3mo', years=15)

    if d.empty and w.empty and m.empty and q.empty:
        st.warning("Keine Daten für die Multi‑TF‑Ansicht gefunden.")
        return

    if not d.empty:
        st.markdown("### Daily")
        st.plotly_chart(_make_tf_figure(d, f"{symbol} – Daily | Candles · EMA5 · BB(20,2) · MA200", show_rsi=show_rsi_tf), use_container_width=True)
    if not w.empty:
        st.markdown("### Weekly")
        st.plotly_chart(_make_tf_figure(w, f"{symbol} – Weekly | Candles · EMA5 · BB(20,2) · MA200", show_rsi=show_rsi_tf), use_container_width=True)
    if not m.empty:
        st.markdown("### Monthly")
        st.plotly_chart(_make_tf_figure(m, f"{symbol} – Monthly | Candles · EMA5 · BB(20,2) · MA200", show_rsi=show_rsi_tf), use_container_width=True)
    if not q.empty:
        st.markdown("### Quarterly")
        st.plotly_chart(_make_tf_figure(q, f"{symbol} – Quarterly | Candles · EMA5 · BB(20,2) · MA200", show_rsi=show_rsi_tf), use_container_width=True)


# --- Multi‑TF Tab ---
with tab_mtf:
    st.subheader("🗓️ Multi‑TF Candles: Daily / Weekly / Monthly / Quarterly")
    render_multi_tf_candles(ticker)

# --- Live 15m Tab -------------------------------------------------------------
with tab_live:
    import time
    from plotly.subplots import make_subplots as _make_subplots_live
    import plotly.graph_objects as _go_live

    st.subheader("⏱️ Live 15-Minuten – Candles, BB(20,2), EMA20, RSI")

    colA, colB, colC = st.columns(3)
    days_back = colA.selectbox("Zeitraum (Tage)", [7, 14, 30], index=2)
    refresh_secs = colB.selectbox("Refresh-Intervall (Sek.)", [30, 60, 120, 300, 900], index=1)
    auto_refresh = colC.toggle("Auto-Refresh", value=True)

    # Local formatter for price ranges for labels
    def _fmt_lvl_live(v: float) -> str:
        try:
            return f"{float(v):,.0f}".replace(",", " ")
        except Exception:
            return str(v)

    df15 = _fetch_15m_df_live(ticker, days=days_back)
    if df15.empty:
        st.info("Keine 15m-Daten verfügbar.")
    else:
        # Two-checkbox block for zones and signals
        cZ, cS = st.columns([1,1])
        show_zones_live = cZ.checkbox("Confluence‑Zonen", value=True)
        show_signals_live = cS.checkbox("15m Entry‑Signale", value=True)

        # --- Zusatz-Controls für Zone Density/Visibility (nach den beiden Checkboxes) ---
        colZ1, colZ2, colZ3 = st.columns([1,1,1])
        max_zones_per_type = int(colZ1.number_input("Max Zonen/Typ", min_value=1, max_value=10, value=3, step=1))
        near_pct = float(colZ2.slider("Nur kursnahe Zonen (±%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1))
        zone_span_opt = colZ3.selectbox("Zonenbreite", ["Vollbreite", "Letzte 5 Tage"], index=1)
        zone_opacity = float(st.slider("Zonen-Deckkraft", min_value=0.0, max_value=0.5, value=0.08, step=0.01))

        # Figure: Price + RSI
        figL = _make_subplots_live(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.82, 0.18])

        # Candles (thicker, high-contrast)
        figL.add_trace(_go_live.Candlestick(
            x=df15.index,
            open=df15['Open'], high=df15['High'], low=df15['Low'], close=df15['Close'],
            name='15m',
            increasing=dict(line=dict(color='lime', width=2.2)),
            decreasing=dict(line=dict(color='red',  width=2.2)),
            whiskerwidth=0.6
        ), row=1, col=1)

        # EMA20 + Bollinger (blau)
        # EMA 5/9/12 für Intraday-Feinsteuerung
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['EMA5'], name='EMA5', line=dict(width=1.6)), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['EMA9'], name='EMA9', line=dict(width=1.6)), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['EMA12'], name='EMA12', line=dict(width=1.6)), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['EMA20'], name='EMA20', line=dict(width=2.0)), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['BB_UP'], name='BB Upper', line=dict(color='#2a5bd7', width=1.2)), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['BB_LO'], name='BB Lower', line=dict(color='#2a5bd7', width=1.2),
                                        fill='tonexty', fillcolor='rgba(42,91,215,0.12)'), row=1, col=1)
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['BB_MID'], name='BB Mid',
                                        line=dict(color='#2a5bd7', width=1.0, dash='dot'), showlegend=False), row=1, col=1)

        # RSI unten
        figL.add_trace(_go_live.Scatter(x=df15.index, y=df15['RSI15'], name='RSI(14) 15m'), row=2, col=1)
        figL.add_hline(y=70, line=dict(color='#888', dash='dot', width=1), row=2, col=1)
        figL.add_hline(y=30, line=dict(color='#888', dash='dot', width=1), row=2, col=1)

        # Theme/Look&Feel
        figL.update_layout(template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                           margin=dict(l=20, r=180, t=50, b=30), font=dict(color='#111111', size=12),
                           hovermode='x unified', dragmode='zoom')
        figL.update_xaxes(showgrid=True, gridcolor='#e5e5e5', row=1, col=1)
        figL.update_yaxes(showgrid=True, gridcolor='#e5e5e5', row=1, col=1)
        figL.update_xaxes(showgrid=True, gridcolor='#e5e5e5', row=2, col=1)
        figL.update_yaxes(showgrid=True, gridcolor='#e5e5e5', row=2, col=1)
        # Make the live chart tall and spacious; preserve right margin for legend
        figL.update_layout(
            height=860,
            margin=dict(l=20, r=200, t=55, b=40),
        )
        figL.update_xaxes(tickfont=dict(size=13, color='#111111'))
        figL.update_yaxes(tickfont=dict(size=13, color='#111111'))

        # Right‑side legend, larger font
        figL.update_layout(
            legend=dict(
                orientation='v',
                font=dict(size=14, color='#111111'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#cfcfcf', borderwidth=1,
                x=1.02, xanchor='left',
                y=0.5,  yanchor='middle'
            )
        )

        # Optional: Confluence‑Zonen (gefiltert & dezent) sowie 15m Entry‑Signale
        long_pts, short_pts = [], []
        zones_live = []
        if show_zones_live:
            try:
                last_price = float(df15['Close'].iloc[-1]) if not df15.empty else None
                # Basisauswahl nach Score
                all_z = [z for z in confluence_zones if z.get('score', 0) >= selected_min_score]
                # Nach Relevanz sortieren: zuerst hoher Score, dann Nähe zum aktuellen Preis
                if last_price is not None:
                    all_z.sort(key=lambda z: (-(z.get('score',0)), abs(z.get('mid', (z['low']+z['high'])/2) - last_price)))
                else:
                    all_z.sort(key=lambda z: (-(z.get('score',0)), z.get('mid', (z['low']+z['high'])/2)))
                # Top‑N je Typ
                buys = [z for z in all_z if z.get('type') == 'buy'][:max_zones_per_type]
                tests = [z for z in all_z if z.get('type') == 'test'][:max_zones_per_type]
                zones_live = buys + tests
                # Kursnähe filtern (optional)
                if last_price is not None and near_pct > 0:
                    zones_live = [z for z in zones_live if abs(z.get('mid', (z['low']+z['high'])/2) - last_price)/max(1.0, last_price) <= (near_pct/100.0)]
                # X‑Ausdehnung der Bänder steuern
                if zone_span_opt == "Letzte 5 Tage":
                    x0 = df15.index[-1] - pd.Timedelta(days=5)
                    x1 = df15.index[-1] + pd.Timedelta(hours=4)
                else:
                    x0 = df15.index[0]
                    x1 = df15.index[-1]
                # Zeichnen – sehr dezentes Fill + dünne Ränder
                for z in zones_live:
                    lo, hi = float(z['low']), float(z['high'])
                    mid = (lo + hi) / 2.0
                    figL.add_shape(
                        type="rect",
                        x0=x0, x1=x1,
                        y0=lo, y1=hi,
                        fillcolor=f"rgba(42,91,215,{zone_opacity})",
                        line=dict(color='#2a5bd7', width=0.8, dash='dot'),
                        layer='below', row=1, col=1
                    )
                    # Range-Label klein an x0 setzen
                    label = f"{'Test' if z.get('type')=='test' else 'Buy'}  {_fmt_lvl_live(lo)}–{_fmt_lvl_live(hi)}  |  S{z.get('score',0)}/5"
                    figL.add_annotation(x=x0, y=mid, text=label,
                                        showarrow=False,
                                        font=dict(size=11, color='#0f1d3a'),
                                        bgcolor='rgba(255,255,255,0.85)',
                                        bordercolor='#2a5bd7', borderwidth=1,
                                        xanchor='left', yanchor='middle', row=1, col=1)
            except Exception:
                pass

        # Auto‑zoom on price range (ignore shapes)
        ymin = float(np.nanmin(df15['Low'])) if 'Low' in df15.columns else float(np.nanmin(df15['Close']))
        ymax = float(np.nanmax(df15['High'])) if 'High' in df15.columns else float(np.nanmax(df15['Close']))
        pad  = max(1.0, (ymax - ymin) * 0.06)
        figL.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=1)

        # 15m Entry‑Signale (RSI + BB Mid Reclaim)
        if show_signals_live and not df15.empty:
            try:
                bb_mid = df15.get('BB_MID')
                rsi15 = df15.get('RSI15')
                last_long_ts = None
                last_short_ts = None
                if bb_mid is not None and rsi15 is not None:
                    # A) Zonen‑basierte Signale (wenn Zonen vorhanden)
                    if len(zones_live) > 0:
                        for ts, row in df15.iterrows():
                            c = float(row.get('Close', float('nan')))
                            if np.isnan(c):
                                continue
                            m = float(bb_mid.loc[ts]) if ts in bb_mid.index and not pd.isna(bb_mid.loc[ts]) else float('nan')
                            r = float(rsi15.loc[ts]) if ts in rsi15.index and not pd.isna(rsi15.loc[ts]) else float('nan')
                            for z in zones_live:
                                lo, hi = float(z['low']), float(z['high'])
                                mid = (lo + hi) / 2.0
                                typ = z.get('type', '')
                                inside = (c >= lo) and (c <= hi)
                                if inside and typ == 'buy' and c >= mid and not np.isnan(r) and r >= 45 and not np.isnan(m) and c > m:
                                    if last_long_ts is None or (ts - last_long_ts) >= pd.Timedelta(minutes=30):
                                        long_pts.append(dict(time=ts, price=c, label=f"15m Long @ {c:.0f} (RSI {r:.0f})"))
                                        last_long_ts = ts
                                if inside and typ == 'test' and c <= mid and not np.isnan(r) and r <= 55 and not np.isnan(m) and c < m:
                                    if last_short_ts is None or (ts - last_short_ts) >= pd.Timedelta(minutes=30):
                                        short_pts.append(dict(time=ts, price=c, label=f"15m Short @ {c:.0f} (RSI {r:.0f})"))
                                        last_short_ts = ts
                    # B) Fallback ohne Zonen: BB_MID Reclaim/Cross + RSI Filter
                    else:
                        prev_c = None; prev_m = None
                        for ts, row in df15.iterrows():
                            c = float(row.get('Close', float('nan')))
                            if np.isnan(c):
                                continue
                            m = float(bb_mid.loc[ts]) if ts in bb_mid.index and not pd.isna(bb_mid.loc[ts]) else float('nan')
                            r = float(rsi15.loc[ts]) if ts in rsi15.index and not pd.isna(rsi15.loc[ts]) else float('nan')
                            if prev_c is not None and prev_m is not None and not np.isnan(m) and not np.isnan(r):
                                # Long: Cross up über BB_MID mit RSI >= 55
                                if prev_c < prev_m and c > m and r >= 55:
                                    if last_long_ts is None or (ts - last_long_ts) >= pd.Timedelta(minutes=30):
                                        long_pts.append(dict(time=ts, price=c, label=f"15m Long @ {c:.0f} (RSI {r:.0f})"))
                                        last_long_ts = ts
                                # Short: Cross down unter BB_MID mit RSI <= 45
                                if prev_c > prev_m and c < m and r <= 45:
                                    if last_short_ts is None or (ts - last_short_ts) >= pd.Timedelta(minutes=30):
                                        short_pts.append(dict(time=ts, price=c, label=f"15m Short @ {c:.0f} (RSI {r:.0f})"))
                                        last_short_ts = ts
                            prev_c, prev_m = c, m
            except Exception:
                long_pts, short_pts = [], []

        # Marker rendern
        if show_signals_live and long_pts:
            figL.add_trace(_go_live.Scatter(
                x=[p['time'] for p in long_pts],
                y=[p['price'] for p in long_pts],
                mode='markers', name='15m Long',
                marker=dict(symbol='triangle-up', size=13, color='green',
                            line=dict(width=1.5, color='white')),
                text=[p['label'] for p in long_pts], hoverinfo='text+x+y'
            ), row=1, col=1)
        if show_signals_live and short_pts:
            figL.add_trace(_go_live.Scatter(
                x=[p['time'] for p in short_pts],
                y=[p['price'] for p in short_pts],
                mode='markers', name='15m Short',
                marker=dict(symbol='triangle-down', size=13, color='red',
                            line=dict(width=1.5, color='white')),
                text=[p['label'] for p in short_pts], hoverinfo='text+x+y'
            ), row=1, col=1)

        # Marker über Shapes zeichnen und nicht clippen
        figL.update_traces(selector=dict(name='15m Long'), cliponaxis=False)
        figL.update_traces(selector=dict(name='15m Short'), cliponaxis=False)

        st.plotly_chart(
            figL,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawrect", "toggleSpikelines"],
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "responsive": True,
            }
        )

    # Simple auto-refresh via session_state timestamp
    if auto_refresh:
        last = st.session_state.get('__live15m_last__', 0.0)
        now = time.time()
        if now - last >= refresh_secs:
            st.session_state['__live15m_last__'] = now
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()


# --- Disziplin-Checkliste Tab (Top-Level) ------------------------------------
with tab_checklist:
    st.subheader("🛡 Trading‑Disziplin‑Checkliste – Gewinne sichern, Verluste begrenzen")

    # Datumsauswahl und State‑Key
    today = pd.to_datetime("today").date()
    sel_date = st.date_input("Datum", value=today, key="disc_date")
    state_key = f"disc_{sel_date.isoformat()}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "prep": {"time_window": False, "max_trades": False, "instruments": False, "setups": False, "written_plan": False},
            "rules": {"loss_limit": False, "gain_goal": False, "fifty_rule": False, "two_strike": False, "no_revenge": False},
            "tech":  {"hard_stops": False, "broker_limits": False, "trail_after_2R": False, "alerts": False},
            "review": {"journal": False, "screens": False, "prep_next": False},
            "params": {"gain_R": 2.0, "loss_R": 1.0}
        }

    disc = st.session_state[state_key]

    # Parameter (R‑Ziele)
    colp1, colp2 = st.columns(2)
    disc["params"]["gain_R"] = colp1.number_input("Tagesgewinnziel (R)", min_value=0.5, max_value=10.0, step=0.5, value=float(disc["params"]["gain_R"]))
    disc["params"]["loss_R"] = colp2.number_input("Tagesverlustlimit (R)", min_value=0.5, max_value=10.0, step=0.5, value=float(disc["params"]["loss_R"]))

    st.markdown("---")
    st.markdown("### 1) Vor dem Handel – Setup & Vorbereitung")
    c1, c2 = st.columns(2)
    disc["prep"]["time_window"]  = c1.checkbox("Handelszeitfenster festgelegt (z. B. 9:30–11:30)", value=disc["prep"]["time_window"])
    disc["prep"]["max_trades"]   = c1.checkbox("Max. Trades pro Tag definiert (3–4)", value=disc["prep"]["max_trades"])
    disc["prep"]["instruments"]  = c2.checkbox("Instrumente/Märkte definiert", value=disc["prep"]["instruments"])
    disc["prep"]["setups"]       = c2.checkbox("Nur vordefinierte Setups handeln", value=disc["prep"]["setups"])
    disc["prep"]["written_plan"] = st.checkbox("Tagesplan schriftlich festgehalten (Bias, Levels, News)", value=disc["prep"]["written_plan"])

    st.markdown("### 2) Während des Handels – Regeln & Limits")
    c3, c4, c5 = st.columns(3)
    disc["rules"]["loss_limit"] = c3.checkbox("Tagesverlustlimit erreicht → Handel beenden", value=disc["rules"]["loss_limit"], help="–1R Regel")
    disc["rules"]["gain_goal"]  = c4.checkbox("Tagesgewinnziel erreicht → Gewinne sichern/Tag beenden", value=disc["rules"]["gain_goal"], help="+2R Regel")
    disc["rules"]["fifty_rule"] = c5.checkbox("50%-Regel aktiv (Gewinn halbiert → Schluss)", value=disc["rules"]["fifty_rule"])
    disc["rules"]["two_strike"] = c3.checkbox("2-Strike-Rule (2 Verluste in Folge → Pause)", value=disc["rules"]["two_strike"])
    disc["rules"]["no_revenge"] = c4.checkbox("Kein Revenge‑Trading (keine Größenerhöhung nach Verlust)", value=disc["rules"]["no_revenge"])

    st.markdown("### 3) Technische Schutzmechanismen – Selbstschutz")
    c6, c7 = st.columns(2)
    disc["tech"]["hard_stops"]     = c6.checkbox("Hard Stops im Markt", value=disc["tech"]["hard_stops"])
    disc["tech"]["broker_limits"]  = c6.checkbox("Broker‑Tageslimits aktiviert (Gewinn & Verlust)", value=disc["tech"]["broker_limits"])
    disc["tech"]["trail_after_2R"] = c7.checkbox("Ab +2R: Stop auf +1R nachziehen (Trailing)", value=disc["tech"]["trail_after_2R"])
    disc["tech"]["alerts"]         = c7.checkbox("Alerts statt Dauerbeobachtung nutzen", value=disc["tech"]["alerts"])

    st.markdown("### 4) Nach dem Handel – Auswertung")
    c8, c9, c10 = st.columns(3)
    disc["review"]["journal"]   = c8.checkbox("Tagesjournal ausgefüllt", value=disc["review"]["journal"])
    disc["review"]["screens"]   = c9.checkbox("Screenshots/Charts markiert", value=disc["review"]["screens"])
    disc["review"]["prep_next"] = c10.checkbox("Nächsten Tag vorbereitet", value=disc["review"]["prep_next"])

    st.markdown("---")
    total_checks = sum([int(v) for grp in [disc["prep"], disc["rules"], disc["tech"], disc["review"]] for v in grp.values()])
    max_checks = 5 + 5 + 4 + 3
    st.progress(total_checks / max_checks)
    st.caption(f"Erledigt: {total_checks} / {max_checks} Checks")

    def _to_markdown(disc):
        lines = []
        lines.append(f"Trading‑Disziplin‑Checkliste – {sel_date}")
        lines.append("")
        lines.append(f"Ziele: Gewinnziel {disc['params']['gain_R']}R, Verlustlimit {disc['params']['loss_R']}R")
        lines.append("")
        def sect(title, items):
            lines.append(title)
            for k, label in items:
                chk = "[x]" if k else "[ ]"
                lines.append(f"- {chk} {label}")
            lines.append("")
        sect("1) Vorbereitung", [
            (disc['prep']['time_window'], "Handelszeitfenster festgelegt"),
            (disc['prep']['max_trades'], "Max. Trades pro Tag definiert"),
            (disc['prep']['instruments'], "Instrumente/Märkte definiert"),
            (disc['prep']['setups'], "Nur vordefinierte Setups"),
            (disc['prep']['written_plan'], "Tagesplan schriftlich festgehalten"),
        ])
        sect("2) Regeln & Limits", [
            (disc['rules']['loss_limit'], "Tagesverlustlimit eingehalten"),
            (disc['rules']['gain_goal'], "Tagesgewinnziel eingehalten"),
            (disc['rules']['fifty_rule'], "50%-Regel angewendet"),
            (disc['rules']['two_strike'], "2‑Strike‑Rule angewendet"),
            (disc['rules']['no_revenge'], "Kein Revenge‑Trading"),
        ])
        sect("3) Technik", [
            (disc['tech']['hard_stops'], "Hard Stops im Markt"),
            (disc['tech']['broker_limits'], "Broker‑Tageslimits aktiviert"),
            (disc['tech']['trail_after_2R'], "Trailing Stop nach +2R"),
            (disc['tech']['alerts'], "Alerts genutzt"),
        ])
        sect("4) Nach dem Handel", [
            (disc['review']['journal'], "Journal ausgefüllt"),
            (disc['review']['screens'], "Screenshots markiert"),
            (disc['review']['prep_next'], "Nächsten Tag vorbereitet"),
        ])
        return "\n".join(lines)

    md_text = _to_markdown(disc)
    st.text_area("Vorschau (Markdown)", md_text, height=240)

    st.download_button(
        "📥 Checkliste als .txt herunterladen",
        data=md_text.encode("utf-8"),
        file_name=f"Disziplin_Checkliste_{sel_date}.txt",
        mime="text/plain"
    )

    if st.button("Reset für diesen Tag"):
        del st.session_state[state_key]
        st.experimental_rerun()


# 🟢 Marktampel
with tab_chart:
    st.subheader("🚦Marktampel – Überblick")
    # Sicheres Auslesen des letzten RSI-Werts
    if not data.empty and 'RSI' in data.columns and not data['RSI'].dropna().empty:
        last_rsi = round(data['RSI'].dropna().iloc[-1], 1)
    else:
        last_rsi = None
    # Sicheres Auslesen der letzten MA50-Werte für die Steigungsberechnung
    if not data.empty and 'MA50' in data.columns and len(data['MA50'].dropna()) >= 5:
        ma_slope = data['MA50'].dropna().iloc[-1] - data['MA50'].dropna().iloc[-5]
    else:
        ma_slope = 0

    # 5-stufige Ampellogik mit klarer Differenzierung
    if last_rsi is not None:
        if last_rsi > 65 and ma_slope > 0.5:
            ampel = "🟢 Sehr bullisch"
        elif last_rsi > 55 and ma_slope > 0:
            ampel = "🟢 Bullisch"
        elif last_rsi > 45:
            ampel = "🟡 Neutral"
        elif last_rsi > 35 or ma_slope < 0:
            ampel = "🟠 Schwach"
        else:
            ampel = "🔴 Sehr schwach"
    else:
        ampel = "⚫ Kein RSI verfügbar"

    # Metriken anzeigen
    st.metric(label="RSI (Letzte Woche)", value=f"{last_rsi}")
    st.metric(label="MA50 Trend (5 Wochen)", value=f"{ma_slope:.1f}")


    # Ampelbeschreibung
    st.markdown(f"**Marktampel:** {ampel}")
    with st.expander("ℹ️ Erläuterung zur Marktampel"):
        st.markdown("""
        Die Marktampel bewertet die aktuelle Marktlage basierend auf dem RSI (Relative Strength Index) sowie dem Trendverlauf des MA50:

        - 🟢 **Sehr bullisch**: RSI &gt; 65 und MA50-Trend deutlich steigend
        - 🟢 **Bullisch**: RSI &gt; 55 und MA50-Trend positiv
        - 🟡 **Neutral**: RSI zwischen 45 und 55
        - 🟠 **Schwach**: RSI unter 45 oder fallender MA50-Trend
        - 🔴 **Sehr schwach**: RSI unter 35 und klar negativer MA50-Trend

        Diese Einschätzung hilft bei der groben Einordnung des Marktumfelds, ersetzt aber keine eigene Analyse.
        """)

# 📥 CSV-Export
export_df = pd.DataFrame({
    'Date': data.index,
    'Close': close_series,
    'RSI': data['RSI'],
    'MA50': data['MA50'],
    'MA200': data['MA200']
    #'Buy_Zone': close_series.index.isin(buy_zone.index),
    #'Test_Zone': close_series.index.isin(test_zone.index)
})
csv = export_df.to_csv(index=False)
#st.download_button("📥 Exportiere Buy-/Test-Zonen als CSV", data=csv, file_name=f'{ticker}_zones.csv', mime='text/csv')

# Debug-Check: Sind Daten vollständig?
if debug_mode:
    st.write(data[['Open', 'High', 'Low', 'Close']].dropna().tail())  # Zeigt letzte 5 Zeilen mit Kursdaten
    st.write(f"Datapoints: {len(data)}")  # Zeigt Anzahl der Zeilen im DataFrame


st.subheader("📊 Interaktiver Chart")
# Prepare buy_signals and test_signals for plotting
plot_df = data.copy()
#plot_df['Buy Signal'] = np.where(plot_df.index.isin(buy_zone.index), plot_df['Close_Series'], np.nan)
#plot_df['Test Signal'] = np.where(plot_df.index.isin(test_zone.index), plot_df['Close_Series'], np.nan)
#buy_signals = plot_df['Buy Signal'].dropna()
#test_signals = plot_df['Test Signal'].dropna()

from plotly.subplots import make_subplots
from datetime import timedelta

# --- Readability helpers ---
def _extend_x_range(fig, df, days: int = 60):
    """Extend x‑axis to the right so labels/annotations are not clipped."""
    if df is None or df.empty:
        return
    x0 = pd.to_datetime(df.index.min())
    x1 = pd.to_datetime(df.index.max()) + pd.Timedelta(days=days)
    # Apply to both rows
    fig.update_xaxes(range=[x0, x1], row=1, col=1)
    fig.update_xaxes(range=[x0, x1], row=2, col=1)


def _add_last_price_labels(fig, df, items):
    """Add right‑side price labels (Close/MA50/MA200 …) with small guide lines."""
    if df is None or df.empty:
        return
    x_last = pd.to_datetime(df.index.max())
    x_label = x_last + pd.Timedelta(days=15)
    for (label, series, color) in items:
        try:
            s = pd.to_numeric(series, errors='coerce').dropna()
            if s.empty:
                continue
            y = float(s.iloc[-1])
            # guide line from last candle to annotation
            fig.add_shape(type="line",
                          x0=x_last, x1=x_label,
                          y0=y, y1=y,
                          line=dict(color=color, width=1, dash='dot'),
                          row=1, col=1)
            fig.add_annotation(x=x_label, y=y,
                               text=f"{label}: {y:.0f}",
                               showarrow=False,
                               font=dict(color="#111111", size=12),
                               bgcolor='rgba(255,255,255,0.95)',
                               bordercolor=color, borderwidth=1,
                               xanchor='left', yanchor='middle',
                               row=1, col=1)
        except Exception:
            continue
fig3 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    row_heights=[0.75, 0.25],
    subplot_titles=(f"{ticker} – Preis (Candlestick, MA50, MA200, Zonen, Fibonacci)", "RSI (14 Tage)")
)
fig3.update_layout(height=1200)

# TradingView-like white theme
fig3.update_layout(
    template='plotly_white',
    paper_bgcolor='#ffffff',
    plot_bgcolor='#ffffff',
    margin=dict(l=70, r=200, t=70, b=60),
    font=dict(size=12, color="#111111"),
    hoverlabel=dict(bgcolor='white'),
    hovermode='x unified',
    dragmode='zoom',
    legend=dict(font=dict(size=11, color="#111111"),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#cfcfcf', borderwidth=1)
)
fig3.update_xaxes(showgrid=True, gridcolor='#e5e5e5', tickfont=dict(size=12, color='#111111'), showline=True, linewidth=1, linecolor='#888', row=1, col=1)
fig3.update_yaxes(showgrid=True, gridcolor='#e5e5e5', tickfont=dict(size=12, color='#111111'), showline=True, linewidth=1, linecolor='#888', tickformat=".0f", row=1, col=1)
fig3.update_xaxes(showgrid=True, gridcolor='#e5e5e5', tickfont=dict(size=12, color='#111111'), showline=True, linewidth=1, linecolor='#888', row=2, col=1)
fig3.update_yaxes(showgrid=True, gridcolor='#e5e5e5', tickfont=dict(size=12, color='#111111'), showline=True, linewidth=1, linecolor='#888', tickformat=".0f", row=2, col=1)
fig3.update_xaxes(rangeslider_visible=False, row=1, col=1)

# TradingView-like crosshair spikes + right-side price scale on main chart
fig3.update_xaxes(showspikes=True, spikemode='across', spikecolor='#777777', spikethickness=1, row=1, col=1)
fig3.update_yaxes(showspikes=True, spikemode='across', spikecolor='#777777', spikethickness=1, row=1, col=1, side='right')
fig3.update_xaxes(showspikes=True, spikemode='across', spikecolor='#777777', spikethickness=1, row=2, col=1)
fig3.update_yaxes(showspikes=True, spikemode='across', spikecolor='#777777', spikethickness=1, row=2, col=1)

# Bedingte Anzeige der Indikatoren (alle in row=1, col=1)
if show_indicators:
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'],  name='MA50',  line=dict(color='orange', width=2.0)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(color='red', width=2.2)), row=1, col=1)

    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA5'],  name='EMA(5)',   line=dict(color='blueviolet', width=1.8)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA9'],  name='EMA(9)',   line=dict(color='goldenrod',  width=1.8)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA14'], name='EMA(14)',  line=dict(color='green',      width=1.8)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA69'], name='EMA(69)',  line=dict(color='magenta',    width=1.8)), row=1, col=1)

    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5W'], name='Weekly EMA(5)', line=dict(color='gray', width=1.5)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5Y'], name='Yearly EMA(5)', line=dict(color='gray', width=1.5), showlegend=False), row=1, col=1)

    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'],  name='MA20',  line=dict(color='red',   width=1.6)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA100'], name='MA100', line=dict(color='brown', width=1.8)), row=1, col=1)

    # Bollinger Bands – TradingView-like blue style with band fill
    fig3.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['BB_upper'],
        name='BB(20,2)',
        line=dict(color='#2a5bd7', width=1.4),
        opacity=1.0
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['BB_lower'],
        showlegend=False,
        line=dict(color='#2a5bd7', width=1.4),
        fill='tonexty',
        fillcolor='rgba(42, 91, 215, 0.12)'
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['BB_mid'],
        name='BB Mid',
        line=dict(color='#2a5bd7', width=1.0, dash='dot'),
        showlegend=False
    ), row=1, col=1)

# # Bedingte Anzeige der Buy/Test Signale (row=1, col=1)
# if show_signals:
#     if not buy_signals.empty:
#         fig3.add_trace(go.Scatter(
#             x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal',
#             marker=dict(symbol='circle', size=10, color='green')), row=1, col=1)
#     if not test_signals.empty:
#         fig3.add_trace(go.Scatter(
#             x=test_signals.index, y=test_signals, mode='markers', name='Test Signal',
#             marker=dict(symbol='x', size=10, color='red')), row=1, col=1)


# Sidebar-Expander für EMA(5)-Kontext
with st.sidebar.expander("EMA(5) – Kontext"):
    st.markdown("""
    **Weekly EMA(5):** Zeigt kurzfristige Trendrichtung im Wochenkontext.  
    **Yearly EMA(5):** Extrem langfristiger Trend, Orientierung bei Makrotrends.  
    Beide Linien helfen bei der Einordnung, ob Buy-/Testzonen im Trend liegen oder konträr sind.
    """)
# Ensure OHLC columns in plot_df for Candlestick
plot_df['Open'] = data['Open']
plot_df['High'] = data['High']
plot_df['Low'] = data['Low']
plot_df['Close'] = data['Close']
# OHLC-Spur ans Ende, damit sie oben liegt

# Falls Spalten ein MultiIndex sind (z. B. durch yfinance bei mehreren Tickers)
if isinstance(plot_df.columns, pd.MultiIndex):
    plot_df.columns = plot_df.columns.get_level_values(0)  # Nur die erste Ebene behalten

plot_df_reset = plot_df.reset_index().rename(columns={plot_df.index.name or 'index': 'Date'})

# Candlestick-Plot in row=1, col=1
fig3.add_trace(go.Candlestick(
    x=plot_df_reset['Date'],
    open=plot_df_reset['Open'],
    high=plot_df_reset['High'],
    low=plot_df_reset['Low'],
    close=plot_df_reset['Close'],
    increasing_line_color='lime',
    decreasing_line_color='red',
    name='Candlestick'
), row=1, col=1)

# Extend x‑axis so annotations/labels on the right remain visible
_extend_x_range(fig3, plot_df, days=90)

# Last price line (TradingView-style)
try:
    last_price_val = float(plot_df_reset['Close'].iloc[-1])
    fig3.add_hline(y=last_price_val, line=dict(color='#222222', width=1, dash='dot'), row=1, col=1)
except Exception:
    pass

# Add last‑price labels for key series (improves readability of current levels)
# Build badge items dynamically based on available columns
badge_items = [("Close", plot_df['Close'], '#111111')]
if 'EMA5' in plot_df.columns:
    badge_items.append(("EMA(5)", plot_df['EMA5'], 'blueviolet'))
if 'MA20' in plot_df.columns:
    badge_items.append(("MA20", plot_df['MA20'], 'red'))
if 'MA50' in plot_df.columns:
    badge_items.append(("MA50", plot_df['MA50'], 'orange'))
if 'MA200' in plot_df.columns:
    badge_items.append(("MA200", plot_df['MA200'], 'red'))
if 'BB_mid' in plot_df.columns:
    badge_items.append(("BB Mid", plot_df['BB_mid'], 'purple'))

_add_last_price_labels(fig3, plot_df, badge_items)


# Helper: pretty format for price levels like 6 200 – 6 300
def _fmt_lvl(v: float) -> str:
    try:
        return f"{float(v):,.0f}".replace(",", " ")
    except Exception:
        return str(v)

# Confluence-Zonen als Bänder (row=1, col=1) – TV-like Blue Style + Range Labels
# Filter nach Mindest-Score und nummerieren je Typ
zones_filtered = [z for z in confluence_zones if z['score'] >= selected_min_score]
# sort by price; buys von unten, tests von oben
buy_z = [z for z in zones_filtered if z.get('type') == 'buy']
Test_z = [z for z in zones_filtered if z.get('type') == 'test']
buy_z.sort(key=lambda z: z['mid'])
Test_z.sort(key=lambda z: z['mid'], reverse=True)

# Numerierung für Label
for idx, z in enumerate(buy_z, 1):
    z['idx'] = idx
for idx, z in enumerate(Test_z, 1):
    z['idx'] = idx

def _draw_zone(z, name_prefix: str):
    # Farben – Blau wie im Beispiel, leichte Variation für Typen
    fill = 'rgba(42, 91, 215, 0.14)' if z.get('type') == 'test' else 'rgba(42, 91, 215, 0.10)'
    border = '#2a5bd7'
    # Band
    fig3.add_shape(
        type="rect",
        x0=plot_df.index[0], x1=plot_df.index[-1],
        y0=z['low'], y1=z['high'],
        fillcolor=fill,
        line=dict(color=border, width=1),
        layer='below', row=1, col=1
    )
    # Outline-Linien
    fig3.add_hline(y=z['low'], line=dict(color=border, width=1, dash='dot'), row=1, col=1)
    fig3.add_hline(y=z['high'], line=dict(color=border, width=1, dash='dot'), row=1, col=1)
    # Range-Text links im Band
    rng = f"{_fmt_lvl(z['low'])}–{_fmt_lvl(z['high'])}"
    label = f"{name_prefix} {z.get('idx', '')}  {rng}"
    fig3.add_annotation(
        x=plot_df.index[int(len(plot_df)*0.02)],
        y=(z['low'] + z['high'])/2,
        text=label,
        showarrow=False,
        font=dict(size=13, color='#0f1d3a'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=border, borderwidth=1,
        xanchor='left', yanchor='middle', row=1, col=1
    )

for z in Test_z:
    _draw_zone(z, "Test Zone")
for z in buy_z:
    _draw_zone(z, "Buy Zone")

# --- Zonen-Validierung unterhalb des Charts ---
with tab_chart:
    st.markdown("---")
    st.markdown("#### 🔎 Zonen-Validierung (Forward-Test)")
    eval_res = evaluate_confluence_zones(data, confluence_zones, lookahead=10)
    c1, c2, c3 = st.columns(3)
    c1.metric("Anzahl getesteter Zonen", f"{eval_res['count']}")
    c2.metric("Trefferquote (±0.5 ATR Ausbruch)", f"{(eval_res['hit_rate']*100):.1f}%" if pd.notna(eval_res['hit_rate']) else "–")
    c3.metric("Ø Vorwärtsrendite (10 Bars)", f"{eval_res['avg_forward_return_%']:.2f}%" if pd.notna(eval_res['avg_forward_return_%']) else "–")
    st.caption("Heuristische Metriken – für robuste Aussage sollten Walk‑Forward‑Backtests pro Regel/Parameter durchgeführt werden.")

rsi_series = data['RSI'].dropna() if 'RSI' in data.columns else pd.Series(dtype=float)
if not rsi_series.empty:
    fig3.add_trace(
        go.Scatter(
            x=rsi_series.index,
            y=rsi_series,
            name='RSI (14)',
            line=dict(color='deepskyblue', width=2)
        ),
        row=2, col=1
    )
    # Add reference lines at 70 and 30, and set y-axis range for RSI subplot
    fig3.add_hline(y=70, line=dict(color='gray', dash='dash'), row=2, col=1)
    fig3.add_hline(y=30, line=dict(color='gray', dash='dash'), row=2, col=1)
    fig3.update_yaxes(title_text="RSI (14 Tage)", range=[0, 100], row=2, col=1)

# --- Chartmuster: Channel Overlay (TradingView-Style) ---

if vereinfachte_trading:
    # Vereinfachte Trading-Ansicht: RSI-Subplot direkt in den Hauptchart integrieren
    from plotly.subplots import make_subplots
    fig3 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} – Preis (Candlestick, MA50, MA200, Zonen, Fibonacci)", "RSI (14)")
    )
    fig3.update_layout(height=900)
    # Candlestick
    fig3.add_trace(go.Candlestick(
        x=plot_df_reset['Date'],
        open=plot_df_reset['Open'],
        high=plot_df_reset['High'],
        low=plot_df_reset['Low'],
        close=plot_df_reset['Close'],
        increasing_line_color='lime',
        decreasing_line_color='red',
        name='Candlestick'
    ), row=1, col=1)
    # MA50, MA200
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], name='MA50', line=dict(dash='dot', color='orange')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(dash='dot', color='red')), row=1, col=1)
    # Confluence Zones als Rechtecke (nur Score >= Schwelle)
    for zone in confluence_zones:
        band_color = "rgba(0, 255, 0, 0.07)" if zone['score'] == 3 else \
                     "rgba(255, 165, 0, 0.07)" if zone['score'] == 2 else \
                     "rgba(128, 128, 128, 0.05)"
        fig3.add_shape(
            type="rect",
            x0=plot_df.index[0], x1=plot_df.index[-1],
            y0=zone['low'], y1=zone['high'],
            fillcolor=band_color,
            line=dict(width=0),
            layer='below',
            row=1, col=1
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=zone['level'],
            text=f"{zone['score']}/3",
            showarrow=False,
            font=dict(size=10, color='#111111'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1,
            xanchor='left',
            row=1, col=1
        )
    # Fibonacci (nur Basis-Levels)
    basic_fibs = ["0.0", "0.236", "0.382", "0.5", "0.618", "0.786", "1.0"]
    for lvl in basic_fibs:
        if lvl in fib:
            val = fib[lvl]
            fig3.add_hline(
                y=val,
                line=dict(dash='dot', color='#555555'),
                opacity=0.3,
                row=1, col=1
            )
            fig3.add_annotation(
                x=plot_df.index[-1],
                y=val,
                text=f"Fib {lvl}",
                showarrow=False,
                font=dict(size=12, color='#444444'),
                bgcolor='rgba(255,255,255,0.85)',
                bordercolor='#555555',
                borderwidth=1,
                xanchor='right',
                row=1, col=1
            )
    # RSI Subplot
    rsi_series = data['RSI'].dropna()
    fig3.add_trace(
        go.Scatter(x=rsi_series.index, y=rsi_series, name='RSI (14)', line=dict(color='deepskyblue', width=2)),
        row=2, col=1
    )
    # RSI Schwellen
    fig3.add_hline(y=float(70), line=dict(color='gray', dash='dash'), row=2, col=1)
    fig3.add_hline(y=float(30), line=dict(color='gray', dash='dash'), row=2, col=1)
    # Layout
    fig3.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#111111"),
        hovermode="x unified",
        title=dict(
            text=f"{ticker} – Vereinfachte Trading-Ansicht",
            x=0.5, xanchor='center',
            font=dict(size=16, color='#111111')
        ),
        height=900
    )
    fig3.update_xaxes(
        gridcolor='#e6e6e6',
        showline=True, linewidth=1, linecolor='#666666',
        showspikes=True, spikecolor="black", spikethickness=1, spikedash='dot',
        rangeslider_visible=False,
        row=1, col=1
    )
    fig3.update_yaxes(
        gridcolor='#e6e6e6',
        showline=True, linewidth=1, linecolor='#666666',
        showspikes=True, spikecolor="black", spikethickness=1, spikedash='dot',
        title_text="Preis",
        row=1, col=1
    )
    fig3.update_yaxes(
        gridcolor='#e6e6e6',
        showline=True, linewidth=1, linecolor='#666666',
        title_text="RSI",
        range=[0, 100],
        row=2, col=1
    )
    # st.plotly_chart(fig3, use_container_width=True)
else:

    # Fibonacci-Extensions in hellgrau
    for lvl, val in fib_ext.items():
        fig3.add_hline(
            y=val,
            line=dict(dash='dot', color='#666666'),
            opacity=0.4
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=val,
            text=f"Ext {lvl}",
            showarrow=False,
            font=dict(size=12, color='#111111'),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='#666666',
            borderwidth=1,
            xanchor='right'
        )

    # Fibonacci-Level in sehr hellem Grau
    for lvl, val in fib.items():
        fig3.add_hline(
            y=val,
            line=dict(dash='dot', color='#555555'),
            opacity=0.3
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=val,
            text=f"Fib {lvl}",
            showarrow=False,
            font=dict(size=12, color='#111111'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#555555',
            borderwidth=1,
            xanchor='right'
        )

# --- Verbesserte Confluence-Zonen-Darstellung mit Rechtecken & Tabelle ---

# Tabelle vorbereiten
zones_table_df = pd.DataFrame([{
    "Type": zone.get('type', ''),
    "Range": f"{_fmt_lvl(zone['low'])}–{_fmt_lvl(zone['high'])}",
    "Level (Mid)": round(zone["mid"], 2),
    "Lower Band": round(zone["low"], 2),
    "Upper Band": round(zone["high"], 2),
    "Score": zone['score']
} for zone in confluence_zones])

# Tabelle anzeigen (automatisch aktualisiert mit Prominenz-Slider)
with tab_chart:
    st.subheader("📄 Übersicht der Confluence Zonen")
    st.dataframe(zones_table_df)

# --- Auto-Tuning Ergebnisse anzeigen -----------------------------------------
with tab_chart:
    if isinstance(st.session_state.get('autotune_best'), dict):
        st.markdown("### 🔬 Auto-Tuning – beste Kombination")
        best = st.session_state['autotune_best']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beste Prominenz", str(best.get('prominence')))
        c2.metric("Min-Score", str(best.get('min_score')))
        c3.metric("Merge-Tol (%)", f"{best.get('merge_tol_pct',0)*100:.2f}")
        c4.metric("Hit-Rate", f"{best.get('hit_rate%', '–')}%")
    if not st.session_state.get('autotune_df', pd.DataFrame()).empty:
        st.dataframe(st.session_state['autotune_df'])

# --- Erweiterung: Profi-Kommentar zur RSI-Interpretation (aus Screenshot) ---
with st.expander("🧠 Profi-Insight: RSI verstehen & Kontext", expanded=False):
    st.markdown("""
    Klassische Parameter wie **RSI**, Stochastic Oscillator oder Williams %R geben wertvolle Hinweise, sind aber oft interpretationsbedürftig.
    Ein hoher RSI muss nicht zwingend zu einer Korrektur führen, er kann auch nur abkühlen, während der Markt konsolidiert oder Seitwärts läuft. 
    Zum Beispiel können andere Sektoren relative Stärke zeigen, sodass der Gesamtindex trotz hohem RSI nicht fällt.

    **Wichtig:** 
    - RSI über 70 ≠ automatisch Short-Signal.
    - RSI sollte immer mit Sektorenrotation, Marktstruktur und Sentiment kombiniert werden.
    - Laut Backtests liefert "immer Short gehen bei RSI > 70" keine profitable Performance ohne korrektes Risiko-Management.

    👉 Nutze RSI lieber als einen Hinweis zur Überhitzung, nicht als alleinigen Trigger.
    """)

# --- Verbesserter LSTM Forecast mit Unsicherheitsband und Reversion ---
st.subheader("🔮 Verbesserter LSTM Forecast mit Unsicherheitsband")
show_lstm = st.checkbox("LSTM Forecast anzeigen", value=False)

if show_lstm:
    st.info("Der Forecast wird mit Unsicherheitsband (±1σ) und Reversionslogik berechnet. Basierend auf Closing, EMA5 und MA20 sowie VIX.")

    # VIX laden
    vix_df = fetch_prices("^VIX", start=start_date, end=end_date, interval="1d")
    vix_df['VIX_SMA5'] = vix_df['Close'].rolling(window=5).mean()
    vix_df['VIX_RSI'] = RSIIndicator(close=vix_df['Close'].squeeze(), window=14).rsi()
    vix_df['VIX_Change'] = vix_df['Close'].pct_change()
    vix_df['Month'] = vix_df.index.month / 12.0

    vix_df = vix_df[['Close', 'VIX_SMA5', 'VIX_RSI', 'VIX_Change', 'Month']]
    vix_df.rename(columns={'Close': 'VIX_Close'}, inplace=True)

    # --- Defensive normalization before join: flatten columns & ensure DatetimeIndex ---
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df = vix_df.copy()
        vix_df.columns = vix_df.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    vix_df.index = pd.to_datetime(vix_df.index)

    features_df = data.join(vix_df, how='left', rsuffix='_vix')
    features_df = features_df[['Close_Series', 'RSI', 'MA50', 'MA20', 'EMA5',
                               'VIX_Close', 'VIX_SMA5', 'VIX_RSI', 'VIX_Change', 'Month']]
    features_df['RSI_Change'] = features_df['RSI'].diff()
    features_df['Close_MA20_Pct'] = (features_df['Close_Series'] - features_df['MA20']) / features_df['MA20']
    features_df['Close_EMA5_Pct'] = (features_df['Close_Series'] - features_df['EMA5']) / features_df['EMA5']
    features_df.dropna(inplace=True)

    features_df = features_df[['Close_Series', 'RSI', 'MA50', 'VIX_Close', 'VIX_SMA5', 'VIX_RSI',
                               'VIX_Change', 'Month', 'RSI_Change', 'Close_MA20_Pct', 'Close_EMA5_Pct']]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)

    def create_sequences(data, seq_len=30):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, 0])
        return np.array(X), np.array(y)

    sequence_length = 30
    X_seq, y_seq = create_sequences(scaled_features, sequence_length)

    import tensorflow as tf
    tf.keras.backend.clear_session()

    model_path = "lstm_model.keras"
    expected_shape = (sequence_length, scaled_features.shape[1])

    if os.path.exists(model_path):
        st.success("✅ Modell gefunden – lade Modell.")
        model = load_model(model_path, compile=False)
        model_shape = model.input_shape[1:]
        if model_shape != expected_shape:
            st.warning(f"⚠️ Shape mismatch! Lösche altes Modell. Expected: {expected_shape}, but was: {model_shape}.")
            os.remove(model_path)
            model = None
        else:
            model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    else:
        model = None

    if model is None:
        st.warning("⚠️ Kein Modell gefunden oder neu erstellt wegen Shape-Wechsel.")
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=expected_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', run_eagerly=True)

    checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    epochs = 50
    batch_size = 16

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.fit(X_seq, y_seq, epochs=1, batch_size=batch_size, verbose=0, callbacks=[checkpoint, early_stop])
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training... Epoche {epoch + 1}/{epochs}")

    progress_bar.progress(1.0)
    status_text.text("✅ Training abgeschlossen.")

    # Forecast-Loop
    last_seq = scaled_features[-sequence_length:].copy()
    forecast_scaled = []
    current_seq = last_seq

    for _ in range(5):
        # --- Defensive shape handling for LSTM input ---
        cs = np.asarray(current_seq)

        # Ensure 2D: (timesteps, features)
        if cs.ndim == 1:
            cs = cs.reshape(-1, 1)

        timesteps = int(cs.shape[0])
        num_features = int(cs.shape[1]) if cs.ndim >= 2 else 0

        # Abort gracefully if features missing or empty sequence
        if num_features == 0 or timesteps == 0:
            st.warning("LSTM: keine Features im aktuellen Sequenzfenster – Vorhersage übersprungen.")
            break

        # Ensure we have at least 'sequence_length' timesteps; use last window
        if timesteps < sequence_length:
            st.warning(f"LSTM: Sequenzlänge {timesteps} < benötigt {sequence_length}. Vorhersage übersprungen.")
            break

        window = cs[-sequence_length:]

        # Predict with inferred feature dimension
        try:
            pred_scaled = model.predict(
                window.reshape(1, sequence_length, num_features),
                verbose=0
            ).squeeze()
        except Exception as _e:
            st.warning(f"LSTM: Prediction fehlgeschlagen: {_e}")
            break

        # Reversion Logic (blend Richtung letztem Wert des ersten Features)
        last_val = float(window[-1, 0])
        pred_scaled = last_val + 0.8 * (float(pred_scaled) - last_val)

        # Neue Zeile: erste Feature-Spalte = Vorhersage, Rest wie letzte Zeile
        new_row = window[-1].copy()
        new_row[0] = pred_scaled

        # Sequenz erweitern und Sliding Window aktualisieren
        cs = np.vstack([cs, new_row])
        current_seq = cs[-sequence_length:]
        forecast_scaled.append(float(pred_scaled))

    # Falls keine gültigen Vorhersagen erzeugt wurden, sauber abbrechen
    if len(forecast_scaled) == 0:
        st.warning("LSTM: keine Vorhersagen erzeugt – bitte Daten/Parameter prüfen (zu wenige Timesteps/Features?).")
    else:
        # Dummy für Inverse Transform
        dummy_zeros = np.zeros((len(forecast_scaled), scaled_features.shape[1]))
        dummy_zeros[:, 0] = np.asarray(forecast_scaled, dtype=float)
        forecast_close = scaler.inverse_transform(dummy_zeros)[:, 0]

        residuals = y_seq - model.predict(X_seq, verbose=0).flatten()
        residual_std = np.std(residuals)

        band_upper = forecast_close + residual_std
        band_lower = forecast_close - residual_std

        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_close,
                                    'Upper': band_upper, 'Lower': band_lower})

        st.subheader("🗒️ Forecast-Tabelle (LSTM)")
        st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}", "Upper": "{:.2f}", "Lower": "{:.2f}"}))

        # --- DB Insert: lstm_forecasts
        params = {
            "seq_length": sequence_length,
            "features": list(features_df.columns),
            "epochs": epochs,
            "batch_size": batch_size
        }
        upsert_forecast(ticker, forecast_df, params)

        # Traces
        forecast_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers',
                                    name='LSTM Forecast', line=dict(color='deepskyblue', width=3))

        upper_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper'], mode='lines',
                                 name='Upper Band', line=dict(color='lightblue', dash='dot'), showlegend=False)

        lower_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower'], mode='lines',
                                 name='Lower Band', line=dict(color='lightblue', dash='dot'), fill='tonexty',
                                 fillcolor='rgba(30, 144, 255, 0.2)', showlegend=False)

        connection_trace = go.Scatter(
            x=[data.index[-1], forecast_df['Date'].iloc[0]],
            y=[data['Close_Series'].iloc[-1], forecast_df['Forecast'].iloc[0]],
            mode='lines', line=dict(color='deepskyblue', dash='dot'), showlegend=False)

        fig3.add_trace(connection_trace)
        fig3.add_trace(upper_trace)
        fig3.add_trace(lower_trace)
        fig3.add_trace(forecast_trace)


with tab_chart:
    st.plotly_chart(
        fig3,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "toggleSpikelines"]
        }
    )





with tab_chart:
    with st.expander("Legende"):
        st.markdown("""
**Linien & Farben**
- **MA50**: dunkelblau (Durchschnitt der letzten 50 Perioden)
- **EMA20**: violett (Exponentieller Durchschnitt, 20 Perioden)
- **Close**: schwarz/grau (Schlusskurs)
- **Bollinger Bands**: mediumpurple
- **Candlesticks**:  
    - **Dunkelgrün**: Bullish (Schlusskurs > Eröffnung)  
    - **Rot**: Bearish (Schlusskurs < Eröffnung)

**Zonen**
- **Confluence Zones**:  
    - **Dunkelgrün**: Score 3/3  
    - **Orange**: Score 2/3  
    - **Grau**: Score 1/3  
  → Je höher der Score, desto mehr Faktoren treffen an dieser Zone zusammen (Preisreaktion, Fibonacci, MA).

**Signale**
- **Grüne Punkte**: Buy-Signal (Kombination aus RSI/MA)
- **Rote Punkte**: Test-Signal (Kombination aus RSI/MA)
        """)

    with st.expander("🧠 Erklärung: Confluence Zones"):
        st.markdown("""
    Die **Confluence Zones** markieren Preisbereiche, an denen mehrere wichtige Faktoren zusammentreffen. Je höher der Score (maximal 3), desto mehr Argumente sprechen für die Relevanz dieser Zone.

    **Bewertungskriterien (je 1 Punkt):**
    1. Lokale Preisreaktion (markantes Hoch oder Tief, Prominenz)
    2. Nähe zu einem Fibonacci-Level
    3. Nähe zu einem gleitenden Durchschnitt (MA200 oder EMA50)

    **Interpretation:**
    - **Score 3/3:** Sehr starke Konfluenz – mehrere wichtige Faktoren treffen zusammen.
    - **Score 2/3:** Mittlere Konfluenz – mindestens zwei Faktoren stimmen überein.
    - **Score 1/3:** Leichte Konfluenz – nur ein Faktor spricht für diese Zone.
        """)
