import math, numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st, yfinance as yf

st.set_page_config(page_title="📈 Portafolios (multi) + Buscador", layout="wide")

# ---------------- Mini base de tickers CL (IPSA-ish) ----------------
# Nota: símbolos pueden cambiar. Esto es guía práctica + validador online.
CL_TICKERS = [
    # ────────────────────── Bancos ──────────────────────
    ("BSANTANDER.SN", "Banco Santander Chile"),
    ("CHILE.SN", "Banco de Chile"),
    ("ITAUCL.SN", "Itau Corpbanca"),
    ("BCI.SN", "BCI"),

    # ─────────────────── Retail / Consumo ───────────────
    ("FALABELLA.SN", "Falabella"),
    ("CENCOSUD.SN", "Cencosud"),
    ("CCU.SN", "CCU"),
    ("PARAUCO.SN", "Parque Arauco"),
    ("RIPLEY.SN", "Ripley"),
    ("ENJOY.SN", "Enjoy"),

    # ─────────────────── Energía y Utilities ─────────────
    ("ENELAM.SN", "Enel Américas"),
    ("ENELCHILE.SN", "Enel Chile"),
    ("COLBUN.SN", "Colbún"),
    ("AESANDES.SN", "AES Andes"),
    ("AGUAS-A.SN", "Aguas Andinas A"),
    ("COPEC.SN", "Empresas Copec"),

    # ─────────────────── Industriales ────────────────────
    ("CAP.SN", "CAP"),
    ("CMPC.SN", "CMPC"),
    ("ENTEL.SN", "Entel"),
    ("INAERIS.SN", "Inversiones Aguas Metropolitanas"),

    # ─────────────────── Materiales / Minería ────────────
    ("SQM-B.SN", "SQM-B"),

    # ─────────────────── Benchmarks CL ───────────────────
    ("^IPSA", "IPSA (Chile)"),
    ("^SPCLXIPSA", "IPSA (alternate)"),

    # ─────────────────── Acciones globales ───────────────
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"),
    ("META", "Meta"),
    ("AMZN", "Amazon"),
    ("TSLA", "Tesla"),

    # ─────────────────── Índices globales ────────────────
    ("^GSPC", "S&P 500"),
    ("^NDX", "NASDAQ 100"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),

    # ─────────────────── ETFs del S&P 500 ────────────────
    ("SPY", "SPDR S&P 500 ETF Trust"),
    ("VOO", "Vanguard S&P 500 ETF"),
    ("IVV", "iShares Core S&P 500 ETF"),
    ("SPLG", "SPDR Portfolio S&P 500 ETF"),
    ("SPYG", "SPDR S&P 500 Growth ETF"),
    ("SPYV", "SPDR S&P 500 Value ETF"),

    # ───────────── ETFs sectoriales (S&P 500) ────────────
    ("XLK", "Technology Select Sector SPDR"),
    ("XLF", "Financial Select Sector SPDR"),
    ("XLE", "Energy Select Sector SPDR"),
    ("XLV", "Health Care Select Sector SPDR"),
    ("XLY", "Consumer Discretionary Select Sector SPDR"),
    ("XLP", "Consumer Staples Select Sector SPDR"),
    ("XLI", "Industrial Select Sector SPDR"),
    ("XLB", "Materials Select Sector SPDR"),
    ("XLRE", "Real Estate Select Sector SPDR"),
    ("XLU", "Utilities Select Sector SPDR"),
    ("XLC", "Communication Services Select Sector SPDR"),

    # ─────────────────── Otros ETFs populares ────────────
    ("QQQ", "Invesco QQQ Trust (NASDAQ 100)"),
    ("VTI", "Vanguard Total Stock Market ETF"),
    ("VO", "Vanguard Mid-Cap ETF"),
    ("VB", "Vanguard Small-Cap ETF"),
    ("VEA", "Vanguard FTSE Developed Markets ETF"),
    ("VWO", "Vanguard FTSE Emerging Markets ETF"),
    ("ARKK", "ARK Innovation ETF"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF"),
    ("IWM", "iShares Russell 2000 ETF"),
    ("SCHD", "Schwab U.S. Dividend Equity ETF"),
]

# ----------------- Utilidades (mejoradas) -----------------

def choose_interval(period, wanted):
    """Asegura combinaciones válidas Yahoo period/interval."""
    if period in ["1mo", "3mo"]:
        return "1d" if wanted not in ["1d", "1wk"] else wanted
    if period in ["6mo", "1y"]:
        return "1d" if wanted not in ["1d", "1wk"] else wanted
    if period in ["2y", "5y", "10y", "max"]:
        return "1wk" if wanted not in ["1d", "1wk", "1mo"] else wanted
    return wanted

@st.cache_data(ttl=3600)
def load_prices(tickers, period=None, interval="1d", start=None, end=None, fill_gaps=True):
    """
    Descarga robusta para 1..N tickers:
    - Soporta period o start/end.
    - Une por OUTER join (no intersección estricta).
    - Rellena huecos por símbolo (ffill) si fill_gaps=True.
    - Devuelve DataFrame de Close por columna.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    frames = []
    for t in tickers:
        sym = t.strip().upper()
        if start and end:
            df = yf.download(sym, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
        else:
            df = yf.download(sym, period=period, interval=interval,
                             auto_adjust=True, progress=False)

        if df.empty:
            continue

        # Normaliza a columna 'Close'
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        s = df["Close"].copy()
        if fill_gaps:
            s = s.replace(0, np.nan).ffill()

        # Asignar nombre de serie (evita error de rename)
        s.name = sym
        frames.append(s)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1, join="outer").sort_index()
    out = out.dropna(how="all")
    return out

def metrics_from_series(series):
    if series.isna().all() or len(series) < 2:
        return dict(ret=np.nan, vol=np.nan, dd=np.nan, sharpe=np.nan, cagr=np.nan)
    rets = series.pct_change().dropna()
    vol_ann = rets.std() * np.sqrt(252)
    roll_max = series.cummax()
    dd = (series/roll_max - 1.0).min()
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252)
    days = (series.index[-1] - series.index[0]).days or 1
    years = days/365.25
    cagr = (series.iloc[-1]/series.iloc[0])**(1/years) - 1 if years>0 else np.nan
    ret_total = series.iloc[-1]/series.iloc[0] - 1.0
    return dict(ret=ret_total, vol=vol_ann, dd=dd, sharpe=sharpe, cagr=cagr)

def build_portfolio_value(prices, tickers, weights, fill_gaps=True):
    """
    - Alinea OUTER, rellena (opcional).
    - Normaliza cada serie por su primer valor válido.
    - Si algún ticker no tiene datos, lo elimina y renormaliza pesos.
    """
    if prices.empty:
        return pd.Series(dtype=float)

    px = prices.copy()
    if fill_gaps:
        px = px.replace(0, np.nan).ffill()

    # Primer valor válido por cada columna
    first_vals = {}
    for c in px.columns:
        idx = px[c].first_valid_index()
        first_vals[c] = px[c].loc[idx] if idx is not None else np.nan
    first = pd.Series(first_vals)

    valid_cols = first[first.notna()].index.tolist()
    if not valid_cols:
        return pd.Series(dtype=float)

    norm = px[valid_cols].divide(first[valid_cols], axis=1)

    # Re-map de pesos a las columnas válidas
    w_map = dict(zip([t.upper() for t in tickers], weights))
    w = np.array([w_map.get(c, 0.0) for c in valid_cols], dtype=float)
    if w.sum() == 0:
        w = np.ones(len(valid_cols)) / len(valid_cols)
    else:
        w = w / w.sum()

    port = (norm * w).sum(axis=1)
    return (port * 100.0).dropna()

def parse_lines_to_weights(txt):
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        tkr, w = parts[0].upper(), float(parts[1])
        rows.append((tkr, w))
    if not rows:
        return [], np.array([])
    tk, w = zip(*rows)
    w = np.array(w, dtype=float)
    w = np.abs(w)
    s = w.sum()
    w = w / (s if s != 0 else 1.0)
    return list(tk), w

def validate_tickers(ticker_list, period, interval):
    """Retorna listas (validos, invalidos) probando que entreguen al menos 1 dato."""
    valids, invalids = [], []
    for t in ticker_list:
        df = load_prices([t], period=period, interval=interval, fill_gaps=True)
        if not df.empty and df.shape[0] > 0:
            valids.append(t)
        else:
            invalids.append(t)
    return valids, invalids

def search_local_catalog(q):
    """Búsqueda simple en la mini base local CL/Global (sin garantía)."""
    q = q.strip().lower()
    if not q:
        return []
    out = []
    for sym, name in CL_TICKERS:
        if q in sym.lower() or q in name.lower():
            out.append((sym, name))
    # evitar duplicados y limitar
    seen, res = set(), []
    for sym, name in out:
        if sym not in seen:
            seen.add(sym); res.append((sym, name))
    return res[:25]

# ----------------- UI global -----------------
st.markdown("<h2 style='margin:0'>📈 Portafolios (múltiples) + Buscador de Tickers</h2>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2,1,1,1])
with c1:
    rango = st.selectbox("Rango", ["3mo","6mo","1y","2y","5y","10y","max"], index=2)
with c2:
    intervalo_raw = st.selectbox("Intervalo", ["1d","1wk","1mo"], index=0)
with c3:
    benchmark = st.text_input("Benchmark (p.ej. ^IPSA, ^GSPC)", "^IPSA")
with c4:
    usar_log = st.toggle("Escala log", value=False)

# Opciones adicionales
opt1, opt2 = st.columns(2)
with opt1:
    use_dates = st.toggle("Usar fechas exactas (start/end)", value=False)
with opt2:
    fill_gaps = st.toggle("Rellenar huecos (ffill)", value=False)

if use_dates:
    d1, d2 = st.columns(2)
    start_date = d1.date_input("Inicio", pd.to_datetime("2018-01-01"))
    end_date   = d2.date_input("Fin",    pd.Timestamp.today())
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
else:
    start_str = end_str = None

intervalo = choose_interval(rango, intervalo_raw)

st.caption("💡 Yahoo Finance vía `yfinance` (gratis). Sufijos .SN para varias acciones chilenas.")
st.divider()

# ----------------- Buscador de tickers -----------------
st.subheader("🔎 Buscador de tickers (catálogo local + validador Yahoo)")
q1, q2 = st.columns([2,1])
with q1:
    query = st.text_input("Busca por nombre o símbolo (ej.: 'santander', 'SQM', 'ENEL')", "")
with q2:
    want_validate = st.toggle("Validar con Yahoo", value=True)

results = search_local_catalog(query) if query else []
if results:
    st.write("Resultados locales:")
    rs_df = pd.DataFrame(results, columns=["Ticker", "Nombre"])
    st.dataframe(rs_df, hide_index=True, use_container_width=True)
    if want_validate:
        syms_to_check = [r[0] for r in results[:12]]
        valid, invalid = validate_tickers(syms_to_check, "6mo", "1d")
        vcol, icol = st.columns(2)
        with vcol:
            st.success(f"Válidos: {', '.join(valid) if valid else '—'}")
        with icol:
            st.warning(f"Sin datos: {', '.join(invalid) if invalid else '—'}")
    st.caption("Copia el Ticker y pégalo en tu cartera abajo (formato: TICKER, peso).")

st.divider()

# ----------------- Múltiples carteras -----------------
st.subheader("🧺 Define múltiples carteras")

# Presets iniciales (editables) — corregido ENELAM
default_portfolios = {
    "Bancos CL": "BSANTANDER.SN, 0.35\nCHILE.SN, 0.35\nITAUCL.SN, 0.30\n",
    "IPSA ejemplo": "FALABELLA.SN, 0.15\nSQM-B.SN, 0.15\nCOPEC.SN, 0.15\nENELAM.SN, 0.15\nCAP.SN, 0.10\nCENCOSUD.SN, 0.10\nBCI.SN, 0.10\nPARAUCO.SN, 0.10\n",
    "USA Tech": "AAPL, 0.30\nMSFT, 0.30\nNVDA, 0.20\nGOOGL, 0.20\n"
}

# ¿Cuántas carteras a la vez?
n = st.number_input("¿Cuántas carteras quieres analizar a la vez?", min_value=1, max_value=6, value=1, step=1)

# Estado
if "portfolios" not in st.session_state:
    st.session_state.portfolios = []
    for name, txt in list(default_portfolios.items())[:3]:
        st.session_state.portfolios.append({"name": name, "text": txt})
while len(st.session_state.portfolios) < n:
    st.session_state.portfolios.append({"name": f"Cartera {len(st.session_state.portfolios)+1}", "text": "# TICKER, peso\n"})
while len(st.session_state.portfolios) > n:
    st.session_state.portfolios.pop()

tabs = st.tabs([f"📊 {p['name']}" for p in st.session_state.portfolios])

all_series = []
for idx, tab in enumerate(tabs):
    with tab:
        coln, colt = st.columns([1,2])
        with coln:
            st.session_state.portfolios[idx]["name"] = st.text_input("Nombre", st.session_state.portfolios[idx]["name"], key=f"name_{idx}")
            if st.button("Validar tickers", key=f"val_{idx}"):
                tks, _ = parse_lines_to_weights(st.session_state.portfolios[idx]["text"])
                v, iv = validate_tickers(tks, "6mo", "1d")
                st.success(f"Válidos: {', '.join(v) if v else '—'}")
                if iv: st.warning(f"Sin datos: {', '.join(iv)}")
        with colt:
            st.session_state.portfolios[idx]["text"] = st.text_area(
                "Define tu cartera (TICKER, peso). Las ponderaciones se normalizan.",
                value=st.session_state.portfolios[idx]["text"],
                height=140,
                key=f"text_{idx}"
            )

        # Parse + data
        tickers, weights = parse_lines_to_weights(st.session_state.portfolios[idx]["text"])
        if not tickers:
            st.info("Agrega al menos una línea 'TICKER, peso'.")
            continue

        # Descarga robusta (period o fechas, con ffill opcional)
        prices = load_prices(
            tickers,
            period=(None if use_dates else rango),
            interval=intervalo,
            start=(start_str if use_dates else None),
            end=(end_str if use_dates else None),
            fill_gaps=fill_gaps
        )

        if prices.empty:
            st.error("No hay datos. Revisa símbolos o rango/intervalo (prueba semanal '1wk' para ≥2y).")
            continue

        port_series = build_portfolio_value(prices, tickers, weights, fill_gaps=fill_gaps)
        if port_series.empty:
            st.error("No fue posible construir la serie del portafolio (sin datos válidos tras alinear).")
            continue

        met = metrics_from_series(port_series)

        # Benchmark
        bench_prices = load_prices(
            [benchmark],
            period=(None if use_dates else rango),
            interval=intervalo,
            start=(start_str if use_dates else None),
            end=(end_str if use_dates else None),
            fill_gaps=fill_gaps
        )
        bench_series = None
        if not bench_prices.empty:
            b = bench_prices.iloc[:,0]
            # Normaliza benchmark a base=100 (usar .loc con first_valid_index)
            fv = b.first_valid_index()
            if fv is not None:
                bench_series = (b / b.loc[fv]) * 100.0

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Retorno", f"{met['ret']*100:,.2f}%")
        k2.metric("Vol anual", f"{met['vol']*100:,.2f}%")
        k3.metric("DD máx", f"{met['dd']*100:,.2f}%")
        k4.metric("Sharpe (≈)", f"{met['sharpe']:.2f}")
        k5.metric("CAGR (≈)", f"{met['cagr']*100:,.2f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_series.index, y=port_series.values,
                                 name=f"{st.session_state.portfolios[idx]['name']} (base=100)", mode="lines"))
        if bench_series is not None:
            fig.add_trace(go.Scatter(x=bench_series.index, y=bench_series.values,
                                     name=f"Benchmark {benchmark}", mode="lines"))
        fig.update_layout(
            template="plotly_dark", height=520,
            xaxis=dict(showgrid=True),
            yaxis=dict(type="log" if usar_log else "linear", title="Índice (base=100)"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10,r=10,t=30,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de pesos
        wdf = pd.DataFrame({"Ticker": tickers, "Peso (norm)": weights}).set_index("Ticker")
        st.dataframe(wdf.style.format({"Peso (norm)":"{:.2%}"}), use_container_width=True)

        # Descargar serie
        st.download_button(
            "⬇️ Descargar serie (CSV)",
            data=port_series.rename("PortfolioIndex").to_csv().encode("utf-8"),
            file_name=f"portfolio_{idx+1}.csv",
            mime="text/csv",
            key=f"dl_{idx}"
        )

        all_series.append((st.session_state.portfolios[idx]["name"], port_series))

# --------- Comparador global (todas las carteras) ---------
st.divider()
st.subheader("🧪 Comparador global (todas las carteras)")
if all_series:
    fig_all = go.Figure()
    for name, s in all_series:
        fig_all.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))

    bp = load_prices(
        [benchmark],
        period=(None if use_dates else rango),
        interval=intervalo,
        start=(start_str if use_dates else None),
        end=(end_str if use_dates else None),
        fill_gaps=fill_gaps
    )
    if not bp.empty:
        bser = bp.iloc[:,0]
        fv = bser.first_valid_index()
        if fv is not None:
            bser = (bser / bser.loc[fv]) * 100.0
            fig_all.add_trace(go.Scatter(x=bser.index, y=bser.values,
                                         name=f"Benchmark {benchmark}", mode="lines"))

    fig_all.update_layout(
        template="plotly_dark", height=520,
        xaxis=dict(showgrid=True),
        yaxis=dict(type="log" if usar_log else "linear", title="Índice (base=100)"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10,r=10,t=30,b=10)
    )
    st.plotly_chart(fig_all, use_container_width=True)
else:
    st.info("Agrega al menos una cartera para comparar.")
