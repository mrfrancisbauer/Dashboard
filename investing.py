# investing.py
# DB-Persistenz & Scoring für den Investing-Tab
from __future__ import annotations
import sqlite3, json
from datetime import datetime, date
import numpy as np
import pandas as pd


# Default DB path (can be overridden via set_db_path)
DB_PATH = "market_dashboard.db"

def set_db_path(db_path: str):
    global DB_PATH
    DB_PATH = db_path
#
# ---------- Index Constituents Fetchers & Bulk Upsert ----------

def _clean_symbol(sym: str) -> str:
    if not isinstance(sym, str):
        return ""
    s = sym.strip().upper()
    # Normalize common unicode chars
    return s.replace("\xa0", "").replace(" ", "")


def fetch_sp500_constituents() -> pd.DataFrame:
    """Fetch S&P 500 from Wikipedia and return normalized df with columns:
    ticker, name, sector, industry, currency, country
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    # Heuristic: first table contains the current list
    df = tables[0].copy()
    # Expected columns: Symbol, Security, GICS Sector, GICS Sub-Industry
    df.rename(columns={
        'Symbol': 'ticker',
        'Security': 'name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'industry'
    }, inplace=True)
    df['ticker'] = df['ticker'].astype(str).map(_clean_symbol)
    df['name'] = df['name'].astype(str)
    df['sector'] = df.get('sector', "").astype(str)
    df['industry'] = df.get('industry', "").astype(str)
    df['currency'] = 'USD'
    df['country'] = 'US'
    return df[['ticker','name','sector','industry','currency','country']].dropna(subset=['ticker'])


def fetch_dax40_constituents() -> pd.DataFrame:
    """Fetch DAX 40 constituents from Wikipedia robustly and normalize to Yahoo (.DE).
    Returns columns: ticker, name, sector, industry, currency, country
    """
    url = "https://en.wikipedia.org/wiki/DAX"
    tables = pd.read_html(url, header=0)

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join([str(x) for x in tup if str(x) != 'nan']).strip() for tup in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    cand = None
    for t in tables:
        t = _flatten_cols(t.copy())
        lower = {c.lower(): c for c in t.columns}
        cols_l = set(lower.keys())
        has_company = any(k in cols_l for k in ["company", "name", "constituent", "constituents"])
        has_ticker  = any(k in cols_l for k in ["ticker", "ticker symbol", "symbol"])
        # Heuristik: brauchbare Größe (>= 30 Zeilen) und beide Felder vorhanden
        if has_company and has_ticker and len(t) >= 30:
            cand = t
            break
    if cand is None:
        # Fallback: nimm die größte Tabelle mit einer 'Company'-artigen Spalte
        sizes = [(len(t), t) for t in tables]
        sizes.sort(key=lambda x: x[0], reverse=True)
        for _, t in sizes:
            t = _flatten_cols(t.copy())
            lower = {c.lower(): c for c in t.columns}
            if any(k in lower for k in ["company", "name", "constituent", "constituents"]):
                cand = t
                break
    if cand is None:
        return pd.DataFrame(columns=["ticker","name","sector","industry","currency","country"])  # empty

    df = _flatten_cols(cand.copy())
    L = {c.lower(): c for c in df.columns}
    def pick(*options):
        for o in options:
            if o in L: return L[o]
        return None

    col_name   = pick("company", "name", "constituent", "constituents")
    col_ticker = pick("ticker", "ticker symbol", "symbol")
    col_sector = pick("sector", "industry", "gics sector", "gics sub-industry")

    if col_ticker is None:
        df["ticker"] = ""
    else:
        df["ticker"] = df[col_ticker].astype(str)

    if col_name is None:
        df["name"] = df.get("Company", df.get("Name", "")).astype(str)
    else:
        df["name"] = df[col_name].astype(str)

    if col_sector is None:
        df["sector"] = ""
    else:
        df["sector"] = df[col_sector].astype(str)

    # Clean / Normalize ticker and enforce .DE
    def _ensure_de(t: str) -> str:
        t = _clean_symbol(t)
        if not t:
            return t
        # manche Einträge haben mehrere Symbole getrennt durch / oder , → erstes nehmen
        for sep in ["/", ",", " "]:
            if sep in t:
                t = t.split(sep)[0]
                break
        if t.endswith('.DE') or '.' in t:
            return t
        return f"{t}.DE"

    df['ticker'] = df['ticker'].astype(str).map(_ensure_de)
    df['industry'] = ''
    df['currency'] = 'EUR'
    df['country']  = 'DE'

    out = df[["ticker","name","sector","industry","currency","country"]].dropna(subset=["ticker"]).drop_duplicates("ticker")
    # Filtere offensichtliche Nicht‑Ticker wie leere Strings
    out = out[out['ticker'].str.len() >= 3]
    return out


def fetch_nasdaq100_constituents() -> pd.DataFrame:
    """Fetch NASDAQ‑100 constituents from Wikipedia robustly.
    Returns columns: ticker, name, sector, industry, currency, country
    """
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url, header=0)

    def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join([str(x) for x in tup if str(x) != 'nan']).strip() for tup in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    cand = None
    for t in tables:
        t = _flatten_cols(t.copy())
        lower = {c.lower(): c for c in t.columns}
        cols_l = set(lower.keys())
        has_company = any(k in cols_l for k in ["company", "name"])
        has_ticker  = any(k in cols_l for k in ["ticker", "symbol"])
        # Heuristik: NASDAQ‑100 Tabelle hat typischerweise >= 80 Zeilen
        if has_company and has_ticker and len(t) >= 80:
            cand = t
            break
    if cand is None:
        # Fallback: größte passende Tabelle
        sizes = [(len(t), t) for t in tables]
        sizes.sort(key=lambda x: x[0], reverse=True)
        for _, t in sizes:
            t = _flatten_cols(t.copy())
            lower = {c.lower(): c for c in t.columns}
            if any(k in lower for k in ["company", "name"]) and any(k in lower for k in ["ticker", "symbol"]):
                cand = t
                break
    if cand is None:
        return pd.DataFrame(columns=["ticker","name","sector","industry","currency","country"])  # empty

    df = _flatten_cols(cand.copy())
    L = {c.lower(): c for c in df.columns}
    def pick(*options):
        for o in options:
            if o in L: return L[o]
        return None

    col_name   = pick("company", "name")
    col_ticker = pick("ticker", "symbol")
    col_sector = pick("sector", "gics sector")
    col_ind    = pick("industry", "gics sub-industry")

    if col_ticker is None:
        df["ticker"] = ""
    else:
        df["ticker"] = df[col_ticker].astype(str)

    if col_name is None:
        df["name"] = df.get("Company", df.get("Name", "")).astype(str)
    else:
        df["name"] = df[col_name].astype(str)

    df['sector']   = df[col_sector].astype(str) if col_sector else ''
    df['industry'] = df[col_ind].astype(str) if col_ind else ''

    # Clean tickers
    def _clean_us(t: str) -> str:
        t = _clean_symbol(t)
        if not t:
            return t
        for sep in ["/", ",", " "]:
            if sep in t:
                t = t.split(sep)[0]
                break
        # Entferne Fußnoten‑Superscripts wie '^', '*' etc., falls vorhanden
        t = t.replace("*", "").replace("^", "")
        return t

    df['ticker'] = df['ticker'].astype(str).map(_clean_us)
    df['currency'] = 'USD'
    df['country']  = 'US'

    out = df[["ticker","name","sector","industry","currency","country"]].dropna(subset=["ticker"]).drop_duplicates("ticker")
    out = out[out['ticker'].str.len() >= 1]
    return out


def populate_equity_universe(db_path: str = DB_PATH) -> dict:
    """Fetch S&P500, DAX40, NASDAQ-100 constituents and upsert into companies.
    Returns dict with counts per index and total.
    """
    ensure_investing_tables(db_path)
    res = {"sp500": 0, "dax40": 0, "nasdaq100": 0, "total": 0}
    frames = []
    try:
        sp = fetch_sp500_constituents(); res["sp500"] = len(sp); frames.append(sp)
    except Exception:
        pass
    try:
        dx = fetch_dax40_constituents(); res["dax40"] = len(dx); frames.append(dx)
    except Exception:
        pass
    try:
        nd = fetch_nasdaq100_constituents(); res["nasdaq100"] = len(nd); frames.append(nd)
    except Exception:
        pass
    if not frames:
        return res
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    # dedupe by ticker (prefer latest occurrence)
    all_df = all_df.dropna(subset=['ticker']).drop_duplicates(subset=['ticker'], keep='last')
    upsert_companies(all_df, db_path=db_path)
    res["total"] = len(all_df)
    return res



# ---------- DDL: Tabellen anlegen (idempotent) ----------
DDL = """
CREATE TABLE IF NOT EXISTS companies (
  ticker   TEXT PRIMARY KEY,
  name     TEXT,
  sector   TEXT,
  industry TEXT,
  currency TEXT,
  country  TEXT
);

CREATE TABLE IF NOT EXISTS fundamentals_ttm (
  ticker            TEXT,
  as_of_date        DATE,
  revenue_ttm       REAL,
  ebit_ttm          REAL,
  eps_ttm           REAL,
  fcf_ttm           REAL,
  gross_margin      REAL,
  op_margin         REAL,
  fcf_margin        REAL,
  roic              REAL,
  roe               REAL,
  net_debt          REAL,
  ebitda_ttm        REAL,
  shares_out        REAL,
  dividend_yield    REAL,
  buyback_yield     REAL,
  interest_coverage REAL,
  capex_ttm         REAL,
  sector            TEXT,
  currency          TEXT,
  PRIMARY KEY (ticker, as_of_date)
);

CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
  ticker     TEXT,
  q_date     DATE,
  revenue    REAL,
  ebit       REAL,
  eps        REAL,
  fcf        REAL,
  shares_out REAL,
  PRIMARY KEY (ticker, q_date)
);
CREATE INDEX IF NOT EXISTS idx_fund_q_ticker ON fundamentals_quarterly(ticker);

CREATE TABLE IF NOT EXISTS market_snapshots (
  ticker           TEXT,
  snap_date        DATE,
  price            REAL,
  market_cap       REAL,
  ev               REAL,
  total_return_3m  REAL,
  total_return_6m  REAL,
  total_return_12m REAL,
  above_ma200      INTEGER,
  PRIMARY KEY (ticker, snap_date)
);

CREATE TABLE IF NOT EXISTS valuation_bands (
  ticker     TEXT,
  metric     TEXT,    -- 'pe','ev_ebit','ev_sales'
  lookback_y INTEGER, -- 5, 10
  p10        REAL,
  p50        REAL,
  p90        REAL,
  as_of_date DATE,
  PRIMARY KEY (ticker, metric, lookback_y, as_of_date)
);

CREATE TABLE IF NOT EXISTS investing_scores (
  ticker          TEXT,
  as_of_date      DATE,
  quality_score   REAL,
  growth_score    REAL,
  value_score     REAL,
  momentum_score  REAL,
  final_rank      REAL,
  style           TEXT,
  passes_quality  INTEGER,
  passes_price    INTEGER,
  PRIMARY KEY (ticker, as_of_date, style)
);
CREATE INDEX IF NOT EXISTS idx_scores_date ON investing_scores(as_of_date);

CREATE TABLE IF NOT EXISTS screen_runs (
  run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at DATETIME,
  params_json TEXT
);

CREATE TABLE IF NOT EXISTS screen_results (
  run_id        INTEGER,
  rank          INTEGER,
  ticker        TEXT,
  quality_score REAL,
  growth_score  REAL,
  value_score   REAL,
  momentum_score REAL,
  final_rank    REAL,
  is_final_pick INTEGER,
  UNIQUE(run_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_screen_results_run ON screen_results(run_id);


"""

def ensure_investing_tables(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(DDL)
    cur = conn.cursor()
    # Watchlist-Tabellen ergänzen
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlists (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist_items (
          watchlist_id INTEGER,
          ticker TEXT,
          PRIMARY KEY (watchlist_id, ticker),
          FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()


# ---------- Helper: Persist historical prices ----------
def upsert_historical_prices(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Upsert daily historical closes into the DB. Expects columns: ticker, date, close."""
    dbp = db_path or DB_PATH
    if df is None or df.empty:
        return
    need = {"ticker", "date", "close"}
    if not need.issubset(set(df.columns)):
        missing = need - set(df.columns)
        raise ValueError(f"upsert_historical_prices: missing columns: {missing}")
    with sqlite3.connect(dbp) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                ticker TEXT,
                date   DATE,
                close  REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        rows = (
            df[["ticker", "date", "close"]]
            .dropna(subset=["ticker", "date", "close"])  # ensure no null keys
            .itertuples(index=False)
        )
        cur.executemany(
            """
            INSERT INTO historical_prices(ticker, date, close)
            VALUES (?,?,?)
            ON CONFLICT(ticker, date) DO UPDATE SET close = excluded.close
            """,
            rows,
        )
        conn.commit()


# ---------- Watchlist-Helper ----------
import sqlite3 as _sqlite3

def _dbp(path: str | None) -> str:
    return path or DB_PATH


def list_watchlists(db_path: str | None = None) -> list[tuple[int, str]]:
    with _sqlite3.connect(_dbp(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM watchlists ORDER BY name")
        return cur.fetchall()


def get_watchlist_tickers(name: str, db_path: str | None = None) -> list[str]:
    if not name:
        return []
    with _sqlite3.connect(_dbp(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM watchlists WHERE name = ?", (name,))
        row = cur.fetchone()
        if not row:
            return []
        wid = row[0]
        cur.execute("SELECT ticker FROM watchlist_items WHERE watchlist_id = ? ORDER BY ticker", (wid,))
        return [r[0] for r in cur.fetchall()]


def upsert_watchlist(name: str, tickers: list[str], db_path: str | None = None) -> int:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not name:
        raise ValueError("Watchlist-Name fehlt")
    with _sqlite3.connect(_dbp(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO watchlists(name) VALUES (?) ON CONFLICT(name) DO UPDATE SET name = excluded.name",
            (name,)
        )
        conn.commit()
        cur.execute("SELECT id FROM watchlists WHERE name = ?", (name,))
        wid = cur.fetchone()[0]
        cur.execute("DELETE FROM watchlist_items WHERE watchlist_id = ?", (wid,))
        cur.executemany(
            "INSERT OR IGNORE INTO watchlist_items(watchlist_id, ticker) VALUES (?,?)",
            [(wid, t) for t in tickers]
        )
        conn.commit()
        return wid


def add_tickers_to_watchlist(name: str, tickers: list[str], db_path: str | None = None) -> int:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    with _sqlite3.connect(_dbp(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO watchlists(name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,))
        conn.commit()
        cur.execute("SELECT id FROM watchlists WHERE name = ?", (name,))
        wid = cur.fetchone()[0]
        cur.executemany(
            "INSERT OR IGNORE INTO watchlist_items(watchlist_id, ticker) VALUES (?,?)",
            [(wid, t) for t in tickers]
        )
        conn.commit()
        return wid


def get_all_company_tickers(db_path: str | None = None) -> list[str]:
    with _sqlite3.connect(_dbp(db_path)) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT DISTINCT ticker FROM companies WHERE ticker IS NOT NULL AND ticker != '' ORDER BY ticker"
            )
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return []


# ---------- Upserts ----------
def upsert_companies(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    if df.empty: return
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    rows = df[['ticker','name','sector','industry','currency','country']].fillna('').itertuples(index=False)
    cur.executemany("""
      INSERT INTO companies(ticker,name,sector,industry,currency,country)
      VALUES(?,?,?,?,?,?)
      ON CONFLICT(ticker) DO UPDATE SET
        name=excluded.name, sector=excluded.sector, industry=excluded.industry,
        currency=excluded.currency, country=excluded.country
    """, rows)
    conn.commit(); conn.close()

def upsert_fundamentals_ttm(df: pd.DataFrame, as_of_date: str, db_path: str = DB_PATH) -> None:
    if df.empty: return
    cols = ['ticker','revenue_ttm','ebit_ttm','eps_ttm','fcf_ttm','gross_margin','op_margin','fcf_margin',
            'roic','roe','net_debt','ebitda_ttm','shares_out','dividend_yield','buyback_yield',
            'interest_coverage','capex_ttm','sector','currency']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    rows = []
    for r in df[cols + []].itertuples(index=False):
        rows.append((r[0], as_of_date, *r[1:]))
    cur.executemany("""
      INSERT INTO fundamentals_ttm
      (ticker,as_of_date,revenue_ttm,ebit_ttm,eps_ttm,fcf_ttm,gross_margin,op_margin,fcf_margin,roic,roe,net_debt,
       ebitda_ttm,shares_out,dividend_yield,buyback_yield,interest_coverage,capex_ttm,sector,currency)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(ticker,as_of_date) DO UPDATE SET
        revenue_ttm=excluded.revenue_ttm, ebit_ttm=excluded.ebit_ttm, eps_ttm=excluded.eps_ttm,
        fcf_ttm=excluded.fcf_ttm, gross_margin=excluded.gross_margin, op_margin=excluded.op_margin,
        fcf_margin=excluded.fcf_margin, roic=excluded.roic, roe=excluded.roe, net_debt=excluded.net_debt,
        ebitda_ttm=excluded.ebitda_ttm, shares_out=excluded.shares_out, dividend_yield=excluded.dividend_yield,
        buyback_yield=excluded.buyback_yield, interest_coverage=excluded.interest_coverage,
        capex_ttm=excluded.capex_ttm, sector=excluded.sector, currency=excluded.currency
    """, rows)
    conn.commit(); conn.close()

def upsert_market_snapshots(df: pd.DataFrame, snap_date: str, db_path: str = DB_PATH):
    """Persist market snapshots. Expected columns: ticker, price, market_cap, ev;
    optional: total_return_3m, total_return_6m, total_return_12m, above_ma200.
    Missing optional fields werden aus historical_prices berechnet.
    """
    dbp = db_path or DB_PATH
    if df is None or df.empty:
        return
    for r in ['ticker', 'price', 'market_cap']:
        if r not in df.columns:
            raise ValueError(f"upsert_market_snapshots: required column '{r}' missing")

    cols = ['ticker','price','market_cap','ev','total_return_3m','total_return_6m','total_return_12m','above_ma200']
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    with sqlite3.connect(dbp) as conn:
        cur = conn.cursor()
        for _, row in df.iterrows():
            tk = str(row['ticker'])
            price = None if pd.isna(row['price']) else float(row['price'])
            mcap = None if pd.isna(row['market_cap']) else float(row['market_cap'])
            ev = None if pd.isna(row['ev']) else float(row['ev'])
            tr3 = None if pd.isna(row['total_return_3m']) else float(row['total_return_3m'])
            tr6 = None if pd.isna(row['total_return_6m']) else float(row['total_return_6m'])
            tr12 = None if pd.isna(row['total_return_12m']) else float(row['total_return_12m'])
            above = None if pd.isna(row['above_ma200']) else float(row['above_ma200'])

            # Fallback: aus historical_prices berechnen
            if any(v is None for v in (tr3, tr6, tr12, above)):
                c3, c6, c12, cabove = _calc_snapshot_metrics_from_db(conn, tk, snap_date)
                tr3 = tr3 if tr3  is not None else c3
                tr6 = tr6 if tr6  is not None else c6
                tr12 = tr12 if tr12 is not None else c12
                above = above if above is not None else cabove

            cur.execute(
                """
                INSERT INTO market_snapshots (
                    ticker, snap_date, price, market_cap, ev,
                    total_return_3m, total_return_6m, total_return_12m, above_ma200
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, snap_date) DO UPDATE SET
                    price = excluded.price,
                    market_cap = excluded.market_cap,
                    ev = excluded.ev,
                    total_return_3m = excluded.total_return_3m,
                    total_return_6m = excluded.total_return_6m,
                    total_return_12m = excluded.total_return_12m,
                    above_ma200 = excluded.above_ma200
                """,
                (tk, snap_date, price, mcap, ev, tr3, tr6, tr12, above)
            )
        conn.commit()

def upsert_scores(df_scores: pd.DataFrame, style: str, as_of_date: str | None = None, db_path: str = DB_PATH) -> None:
    if df_scores.empty: return
    as_of_date = as_of_date or datetime.utcnow().date().isoformat()
    need = ['ticker','quality_score','growth_score','value_score','momentum_score','final_rank','passes_quality','passes_price']
    for c in need:
        if c not in df_scores.columns: df_scores[c] = np.nan
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    rows = []
    for r in df_scores[need].itertuples(index=False):
        rows.append((r[0], as_of_date, r[1], r[2], r[3], r[4], r[5], style, int(r[6]) if not pd.isna(r[6]) else 0, int(r[7]) if not pd.isna(r[7]) else 0))
    cur.executemany("""
      INSERT INTO investing_scores
      (ticker,as_of_date,quality_score,growth_score,value_score,momentum_score,final_rank,style,passes_quality,passes_price)
      VALUES (?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(ticker,as_of_date,style) DO UPDATE SET
        quality_score=excluded.quality_score, growth_score=excluded.growth_score,
        value_score=excluded.value_score, momentum_score=excluded.momentum_score,
        final_rank=excluded.final_rank, passes_quality=excluded.passes_quality,
        passes_price=excluded.passes_price
    """, rows)
    conn.commit(); conn.close()

def save_screen_results(params_dict: dict, df_ranked: pd.DataFrame, db_path: str = DB_PATH) -> int:
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    cur.execute("INSERT INTO screen_runs(started_at, params_json) VALUES(?,?)",
                (datetime.utcnow().isoformat(timespec='seconds'), json.dumps(params_dict, ensure_ascii=False)))
    run_id = cur.lastrowid
    payload = []
    for i, r in enumerate(df_ranked.itertuples(index=False), start=1):
        payload.append((run_id, i, getattr(r, 'ticker'), getattr(r, 'quality_score', None),
                        getattr(r, 'growth_score', None), getattr(r, 'value_score', None),
                        getattr(r, 'momentum_score', None), getattr(r, 'final_rank', None),
                        int(getattr(r, 'final_pick', 0))))
    cur.executemany("""
      INSERT INTO screen_results(run_id,rank,ticker,quality_score,growth_score,value_score,momentum_score,final_rank,is_final_pick)
      VALUES (?,?,?,?,?,?,?,?,?)
    """, payload)
    conn.commit(); conn.close()
    return run_id


# ---------- Laden letzter Run ----------
def load_last_screen_results(db_path: str = DB_PATH) -> tuple[int | None, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    run = cur.execute("SELECT run_id, started_at, params_json FROM screen_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if not run:
        conn.close()
        return None, pd.DataFrame()
    run_id = run[0]
    df = pd.read_sql_query("SELECT * FROM screen_results WHERE run_id = ? ORDER BY rank ASC", conn, params=(run_id,))
    conn.close()
    return run_id, df


# ---------- Scoring (sektor-neutral) ----------
def _zscore_sector(df: pd.DataFrame, col: str, sector_col: str = "sector") -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    def _z(s):
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std): return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    return df.groupby(sector_col)[col].transform(_z)

def compute_investing_scores(df: pd.DataFrame, style: str = "garp") -> pd.DataFrame:
    df = df.copy()

    # invers & derived
    df['inv_ev_ebit'] = 1.0 / df['ev_ebit'].replace({0: np.nan}) if 'ev_ebit' in df.columns else np.nan
    df['inv_pe']      = 1.0 / df['pe'].replace({0: np.nan})       if 'pe' in df.columns else np.nan
    df['neg_net_debt_ebitda'] = -df['net_debt_ebitda'] if 'net_debt_ebitda' in df.columns else np.nan

    # Value
    val_raw = (
        0.35 * _zscore_sector(df, 'inv_ev_ebit') +
        0.35 * _zscore_sector(df, 'inv_pe') +
        0.30 * _zscore_sector(df, 'fcf_yield')
    )
    df['value_score'] = (50 + 10 * val_raw.clip(-5, 5)).clip(0, 100)

    # Quality
    qual_raw = (
        0.30 * _zscore_sector(df, 'roic') +
        0.20 * _zscore_sector(df, 'fcf_margin') +
        0.20 * _zscore_sector(df, 'interest_coverage') +
        0.20 * _zscore_sector(df, 'piotroski_f') +
        0.10 * _zscore_sector(df, 'neg_net_debt_ebitda')
    )
    df['quality_score'] = (50 + 10 * qual_raw.clip(-5, 5)).clip(0, 100)

    # Growth
    grow_raw = (
        0.40 * _zscore_sector(df, 'sales_cagr') +
        0.35 * _zscore_sector(df, 'fcf_cagr') +
        0.15 * _zscore_sector(df, 'eps_cagr') +
        0.10 * _zscore_sector(df, 'rule_of_40')
    )
    df['growth_score'] = (50 + 10 * grow_raw.clip(-5, 5)).clip(0, 100)

    # Momentum
    mom_raw = (
        0.35 * _zscore_sector(df, 'total_return_12m') +
        0.35 * _zscore_sector(df, 'total_return_6m') +
        0.20 * _zscore_sector(df, 'total_return_3m') +
        0.10 * _zscore_sector(df, 'above_ma200')
    )
    df['momentum_score'] = (50 + 10 * mom_raw.clip(-5, 5)).clip(0, 100)

    # Final
    style = (style or 'garp').lower()
    if style == 'value':
        w = dict(q=0.20, g=0.10, v=0.55, m=0.15)
    elif style == 'dividend':
        w = dict(q=0.30, g=0.10, v=0.40, m=0.20)
    else:
        w = dict(q=0.40, g=0.30, v=0.20, m=0.10)
    df['final_rank'] = (w['q']*df['quality_score'] + w['g']*df['growth_score'] +
                        w['v']*df['value_score'] + w['m']*df['momentum_score'])

    df['passes_quality'] = (df['quality_score'] >= 60).fillna(False)
    df['passes_price'] = (df['value_score'] >= 60).fillna(False)
    return df

# ---------- End-to-end Persistieren eines Screens ----------
def persist_investing_run(df: pd.DataFrame, params: dict, style: str = "garp", db_path: str = DB_PATH) -> int:
    """
    Erwartet ein DataFrame mit mindestens:
    ['ticker','sector','price','market_cap','ev','total_return_3m','total_return_6m','total_return_12m','above_ma200']
    sowie die Fundamentals-Spalten, die compute_investing_scores nutzt.
    """
    as_of_date = datetime.utcnow().date().isoformat()
    scored = compute_investing_scores(df, style=style)

    # Scores speichern
    upsert_scores(
        scored[['ticker','quality_score','growth_score','value_score','momentum_score','final_rank','passes_quality','passes_price']],
        style=style, as_of_date=as_of_date, db_path=db_path
    )

    # Market Snapshot speichern (falls vorhanden)
    ms_cols = ['ticker','price','market_cap','ev','total_return_3m','total_return_6m','total_return_12m','above_ma200']
    if set(ms_cols).issubset(scored.columns):
        upsert_market_snapshots(scored[ms_cols], snap_date=as_of_date, db_path=db_path)

    # Ergebnistabelle
    ranked = scored.sort_values('final_rank', ascending=False).reset_index(drop=True)
    ranked['final_pick'] = (ranked['passes_quality'] & ranked['passes_price']).astype(int)
    run_id = save_screen_results(params, ranked, db_path=db_path)
    return run_id

def _calc_snapshot_metrics_from_db(conn: sqlite3.Connection, ticker: str, snap_date: str) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute 3/6/12M total returns and above_ma200 from historical_prices for a given ticker and snap_date.
    Returns (tr_3m, tr_6m, tr_12m, above_ma200).
    """
    try:
        q = ("SELECT date, close FROM historical_prices WHERE ticker = ? AND date <= ? ORDER BY date ASC")
        h = pd.read_sql(q, conn, params=(ticker, snap_date))
        if h.empty:
            return None, None, None, None
        h['date'] = pd.to_datetime(h['date'])
        h.set_index('date', inplace=True)
        c = pd.to_numeric(h['close'], errors='coerce').dropna()
        if c.empty:
            return None, None, None, None

        def _ret(months: int):
            if len(c) < 2:
                return None
            target_date = pd.to_datetime(snap_date) - pd.DateOffset(months=months)
            c_past = c.loc[:target_date]
            if c_past.empty:
                return None
            past = float(c_past.iloc[-1])
            last = float(c.iloc[-1])
            if past <= 0:
                return None
            return (last / past) - 1.0

        tr3  = _ret(3)
        tr6  = _ret(6)
        tr12 = _ret(12)

        ma200 = c.rolling(200).mean()
        above = None
        if ma200.dropna().size:
            above = float(c.iloc[-1] > ma200.iloc[-1])
        return tr3, tr6, tr12, above
    except Exception:
        return None, None, None, None

import plotly.graph_objects as go
import yfinance as _yf

def _to_float_series(s):
    import pandas as _pd
    try:
        ss = _pd.to_numeric(_pd.Series(s).dropna(), errors='coerce')
        ss = ss[~ss.isna()]
        return ss.astype(float)
    except Exception:
        return _pd.Series(dtype='float64')

def _load_fundamentals_yf(tk: str):
    t = _yf.Ticker(tk)
    fin = t.financials.copy() if hasattr(t, 'financials') else None
    bs  = t.balance_sheet.copy() if hasattr(t, 'balance_sheet') else None
    cf  = t.cashflow.copy() if hasattr(t, 'cashflow') else None
    info = t.info if hasattr(t, 'info') else {}
    return fin, bs, cf, info

def _extract_series(fin, bs, cf):
    import pandas as _pd
    def _row(df, names):
        if df is None or df.empty:
            return _pd.Series(dtype='float64')
        for n in names:
            if n in df.index:
                return _to_float_series(df.loc[n]).sort_index()
        return _pd.Series(dtype='float64')

    revenue      = _row(fin, ["Total Revenue"])
    net_income   = _row(fin, ["Net Income"])
    gross_profit = _row(fin, ["Gross Profit"])
    op_income    = _row(fin, ["Operating Income"])

    total_equity = _row(bs, ["Total Stockholder Equity"])
    total_debt   = _row(bs, ["Total Debt"])
    cash         = _row(bs, ["Cash And Cash Equivalents"])

    op_cf  = _row(cf, ["Operating Cash Flow"])
    capex  = _row(cf, ["Capital Expenditure"]) * -1.0
    div_pay= _row(cf, ["Dividends Paid"]).abs()

    fcf = op_cf.add(capex, fill_value=0)

    # Margins
    gm = (gross_profit / revenue)
    om = (op_income / revenue)
    nm = (net_income / revenue)

    dte = (total_debt / total_equity)

    return {
        'revenue': revenue,
        'net_income': net_income,
        'gross_margin': gm,
        'oper_margin': om,
        'net_margin': nm,
        'fcf': fcf,
        'dividends': div_pay,
        'total_debt': total_debt,
        'total_equity': total_equity,
        'dte': dte,
        'cash': cash,
    }

def _load_price_history(tk: str, years: int = 10):
    import pandas as _pd
    df = _yf.download(tk, period=f"{years}y", interval='1d', auto_adjust=False, progress=False)
    if df is None or df.empty:
        return _pd.DataFrame()
    if isinstance(df.columns, _pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = _pd.to_datetime(df.index)
    return df.sort_index().dropna()

def render_investing_analysis(ticker: str, benchmark: str = "SPY"):
    import streamlit as _st
    import pandas as _pd
    _st.markdown(f"### 🧪 Analyse: {ticker}")

    fin, bs, cf, info = _load_fundamentals_yf(ticker)
    series = _extract_series(fin, bs, cf)

    # Revenue & Net Income
    try:
        rev = series['revenue']; ni = series['net_income']
        if not rev.empty:
            fig = go.Figure()
            fig.add_bar(x=rev.index, y=rev.values, name='Revenue', opacity=0.85)
            if not ni.empty:
                fig.add_trace(go.Scatter(x=ni.index, y=ni.values, name='Net Income', mode='lines+markers'))
            fig.update_layout(template='plotly_white', height=380, title='Revenue & Net Income (annual)')
            _st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Margins
    try:
        gm = series['gross_margin']; om = series['oper_margin']; nm = series['net_margin']
        if any([not s.empty for s in [gm, om, nm]]):
            fig = go.Figure()
            if not gm.empty: fig.add_trace(go.Scatter(x=gm.index, y=gm.values*100, name='Gross %', mode='lines+markers'))
            if not om.empty: fig.add_trace(go.Scatter(x=om.index, y=om.values*100, name='Operating %', mode='lines+markers'))
            if not nm.empty: fig.add_trace(go.Scatter(x=nm.index, y=nm.values*100, name='Net %', mode='lines+markers'))
            fig.update_layout(template='plotly_white', height=320, title='Margins (annual, %)', yaxis_title='%')
            _st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # FCF vs Dividends
    try:
        fcf = series['fcf']; divs = series['dividends']
        if not fcf.empty or not divs.empty:
            idx = sorted(set(fcf.index).union(set(divs.index)))
            fcf_a = fcf.reindex(idx); div_a = divs.reindex(idx)
            fig = go.Figure()
            fig.add_bar(x=idx, y=fcf_a.values, name='Free Cash Flow')
            fig.add_bar(x=idx, y=div_a.values, name='Dividends Paid')
            fig.update_layout(barmode='group', template='plotly_white', height=320, title='FCF vs Dividends (annual)')
            _st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Debt / Equity
    try:
        dte = series['dte']
        if not dte.empty:
            fig = go.Figure()
            fig.add_bar(x=dte.index, y=dte.values, name='Debt/Equity')
            fig.update_layout(template='plotly_white', height=300, title='Debt / Equity (annual)')
            _st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Relative & Risk
    _st.markdown("### ⚖️ Relative & Risiko")
    px = _load_price_history(ticker, years=10)
    bench_px = _load_price_history(benchmark, years=10)
    if not px.empty:
        # RS vs Benchmark
        try:
            if not bench_px.empty:
                rs = (px['Close'] / px['Close'].iloc[0]) / (bench_px['Close'] / bench_px['Close'].iloc[0])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name=f'RS vs {benchmark}', mode='lines'))
                fig.update_layout(template='plotly_white', height=320, title=f'Relative Strength vs {benchmark}')
                _st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
        # Drawdown
        try:
            c = px['Close'].astype(float)
            cummax = c.cummax()
            dd = (c / cummax - 1.0) * 100.0
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name='Drawdown %', mode='lines'))
            fig.update_layout(template='plotly_white', height=280, title='Drawdown (%)', yaxis_title='%')
            _st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
