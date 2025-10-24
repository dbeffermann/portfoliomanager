import math, numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st, yfinance as yf
import json, base64

st.set_page_config(page_title="📈 Portafolios Manager", layout="wide")

# ===== Funciones de persistencia =====
def _encode_snapshot_to_url(snap: dict) -> str:
    raw = json.dumps(snap, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")

def _decode_snapshot_from_url(b64: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(b64.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}

def _apply_state_snapshot(snap: dict):
    """Aplica un snapshot al estado actual (defensivo)."""
    try:
        st.session_state._restore_guard = True
        if "rango" in snap and st.session_state.get("rango") != snap["rango"]:
            st.session_state["rango"] = snap["rango"]
        if "intervalo" in snap and st.session_state.get("intervalo_raw") != snap["intervalo"]:
            st.session_state["intervalo_raw"] = snap["intervalo"]
        if "benchmark" in snap and st.session_state.get("benchmark") != snap["benchmark"]:
            st.session_state["benchmark"] = snap["benchmark"]
        if "use_dates" in snap and st.session_state.get("use_dates") != snap["use_dates"]:
            st.session_state["use_dates"] = snap["use_dates"]
        if "fill_gaps" in snap and st.session_state.get("fill_gaps") != snap["fill_gaps"]:
            st.session_state["fill_gaps"] = snap["fill_gaps"]
        if snap.get("use_dates", False):
            if snap.get("start") and st.session_state.get("start_str") != snap["start"]:
                st.session_state["start_str"] = snap["start"]
            if snap.get("end") and st.session_state.get("end_str") != snap["end"]:
                st.session_state["end_str"] = snap["end"]
        if snap.get("portfolios"):
            current_portfolios = st.session_state.get("portfolios", [])
            if len(current_portfolios) != len(snap["portfolios"]) or \
               any(current_portfolios[i].get("text", "") != snap["portfolios"][i].get("text", "") 
                   for i in range(min(len(current_portfolios), len(snap["portfolios"])))):
                st.session_state.portfolios = []
                for item in snap["portfolios"][:6]:
                    st.session_state.portfolios.append({
                        "name": item.get("name", "Cartera"),
                        "text": item.get("text", "# TICKER, peso\n")
                    })
    finally:
        st.session_state._restore_guard = False

params = st.query_params
if "s" in params and not st.session_state.get("_initialized", False):
    snap = _decode_snapshot_from_url(params["s"])
    if "portfolios" not in st.session_state:
        st.session_state.portfolios = []
    _apply_state_snapshot(snap)
    st.session_state._initialized = True

# ===== Base de tickers =====
CL_TICKERS = [
    # Bancos
    ("BSANTANDER.SN", "Banco Santander Chile"),
    ("CHILE.SN", "Banco de Chile"),
    ("ITAUCL.SN", "Itau Corpbanca"),
    ("BCI.SN", "BCI"),
    # Retail
    ("FALABELLA.SN", "Falabella"),
    ("CENCOSUD.SN", "Cencosud"),
    ("CCU.SN", "CCU"),
    # Energía
    ("ENELAM.SN", "Enel Américas"),
    ("ENELCHILE.SN", "Enel Chile"),
    ("COLBUN.SN", "Colbún"),
    ("COPEC.SN", "Empresas Copec"),
    # Industriales
    ("CAP.SN", "CAP"),
    ("CMPC.SN", "CMPC"),
    ("ENTEL.SN", "Entel"),
    # Minería
    ("SQM-B.SN", "SQM-B"),
    # Benchmarks Chile
    ("ECH", "iShares MSCI Chile ETF"),
    ("ILF", "iShares Latin America 40 ETF"),
    ("^IPSA", "IPSA (Chile) - datos limitados"),
    # Acciones globales
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"),
    ("META", "Meta"),
    ("AMZN", "Amazon"),
    ("TSLA", "Tesla"),
    # Índices globales
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones"),
    # ETFs
    ("SPY", "SPDR S&P 500 ETF"),
    ("QQQ", "Invesco QQQ Trust"),
    ("VTI", "Vanguard Total Stock Market"),
    ("VEA", "Vanguard FTSE Developed Markets"),
    ("VWO", "Vanguard FTSE Emerging Markets"),
]

# ===== Utilidades =====
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
    """Descarga robusta para 1..N tickers."""
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

        # Verificar que tengamos suficientes datos
        min_expected_days = 5 if period in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"] else 1
        if len(df) < min_expected_days:
            continue

        # Manejar estructura de columnas (MultiIndex vs simples)
        if len(df.columns.names) > 1:  # MultiIndex
            close_col = None
            for col in df.columns:
                if 'Close' in str(col):
                    close_col = col
                    break
            if close_col is None:
                continue
            s = df[close_col].copy()
        else:
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            elif "Close" not in df.columns:
                continue
            s = df["Close"].copy()

        if fill_gaps:
            s = s.replace(0, np.nan).ffill()

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
    """Construye valor del portafolio."""
    if prices.empty:
        return pd.Series(dtype=float)

    px = prices.copy()
    if fill_gaps:
        px = px.replace(0, np.nan).ffill()

    # Normalizar por primer valor válido
    first_vals = {}
    for c in px.columns:
        idx = px[c].first_valid_index()
        first_vals[c] = px[c].loc[idx] if idx is not None else np.nan
    first = pd.Series(first_vals)

    valid_cols = first[first.notna()].index.tolist()
    if not valid_cols:
        return pd.Series(dtype=float)

    norm = px[valid_cols].divide(first[valid_cols], axis=1)

    # Re-map de pesos
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

def search_local_catalog(q):
    """Búsqueda simple en la base local."""
    q = q.strip().lower()
    if not q:
        return []
    out = []
    for sym, name in CL_TICKERS:
        if q in sym.lower() or q in name.lower():
            out.append((sym, name))
    seen, res = set(), []
    for sym, name in out:
        if sym not in seen:
            seen.add(sym); res.append((sym, name))
    return res[:25]

def _get_state_snapshot():
    """Snapshot del estado del usuario."""
    return {
        "rango": st.session_state.get("rango", "1y"),
        "intervalo": st.session_state.get("intervalo_raw", "1d"),
        "benchmark": st.session_state.get("benchmark", "ECH"),
        "use_dates": st.session_state.get("use_dates", False),
        "start": st.session_state.get("start_str", "") or "",
        "end": st.session_state.get("end_str", "") or "",
        "fill_gaps": st.session_state.get("fill_gaps", True),
        "portfolios": [
            {"name": p["name"], "text": p["text"]}
            for p in st.session_state.get("portfolios", [])
        ],
    }

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## 💾 Persistencia")
    if st.button("🔗 Guardar en URL"):
        b64 = _encode_snapshot_to_url(_get_state_snapshot())
        st.query_params["s"] = b64
        st.success("Estado guardado en la URL")
    
    if st.button("🔄 Resetear", type="secondary"):
        st.query_params.clear()
        keys_to_remove = [k for k in st.session_state.keys() if k not in ["_restore_guard", "_initialized"]]
        for key in keys_to_remove:
            del st.session_state[key]
        st.session_state._initialized = False
        st.rerun()

    snap_now = _get_state_snapshot()
    st.download_button(
        "⬇️ Exportar JSON",
        data=json.dumps(snap_now, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="portfolios_state.json",
        mime="application/json"
    )

    up = st.file_uploader("⬆️ Importar JSON", type=["json"])
    if up is not None:
        try:
            imported = json.loads(up.read().decode("utf-8"))
            st.query_params.clear()
            _apply_state_snapshot(imported)
            st.session_state._initialized = True
            st.success("Estado importado")
            st.rerun()
        except Exception:
            st.error("Archivo inválido")

# ===== UI PRINCIPAL =====
st.markdown("<h1 style='margin:0'>📈 Portafolios Manager</h1>", unsafe_allow_html=True)

# Controles principales con help
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
with col1:
    rango_options = ["3mo","6mo","1y","2y","5y","10y","max"]
    current_rango = st.session_state.get("rango", "1y")
    rango_index = rango_options.index(current_rango) if current_rango in rango_options else 2
    rango = st.selectbox(
        "Período de tiempo", 
        rango_options, 
        index=rango_index, 
        key="rango",
        help="Período histórico para analizar. Usa 1y-2y para análisis general, 5y-max para tendencias de largo plazo"
    )
with col2:
    intervalo_raw = st.selectbox(
        "Frecuencia", 
        ["1d","1wk","1mo"], 
        index=0, 
        key="intervalo_raw",
        help="1d = datos diarios (más detalle), 1wk = semanales (menos ruido), 1mo = mensuales (tendencias)"
    )
with col3:
    benchmark = st.text_input(
        "Benchmark", 
        st.session_state.get("benchmark", "ECH"), 
        key="benchmark",
        help="Índice de referencia. Ejemplos: ECH (Chile), ^GSPC (S&P 500), SPY (ETF S&P 500)"
    )
with col4:
    usar_log = st.toggle(
        "Escala log", 
        value=False, 
        key="usar_log",
        help="Escala logarítmica facilita comparar activos con grandes diferencias de precio"
    )

# Opciones adicionales
opt1, opt2 = st.columns(2)
with opt1:
    use_dates = st.toggle(
        "Fechas personalizadas", 
        value=st.session_state.get("use_dates", False), 
        key="use_dates",
        help="Permite seleccionar fechas específicas en lugar de períodos predefinidos"
    )
with opt2:
    fill_gaps = st.toggle(
        "Rellenar huecos", 
        value=st.session_state.get("fill_gaps", True), 
        key="fill_gaps",
        help="Completa datos faltantes usando el último precio válido (forward fill)"
    )

if use_dates:
    d1, d2 = st.columns(2)
    default_start = pd.to_datetime(st.session_state.get("start_str", "2018-01-01"))
    default_end = pd.to_datetime(st.session_state.get("end_str", pd.Timestamp.today().strftime("%Y-%m-%d")))
    start_date = d1.date_input("Fecha inicio", default_start)
    end_date = d2.date_input("Fecha fin", default_end)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    st.session_state["start_str"] = start_str
    st.session_state["end_str"] = end_str
else:
    start_str = end_str = None

intervalo = choose_interval(rango, intervalo_raw)

st.info("💡 **Los gráficos muestran índice base 100** (rendimiento relativo desde el inicio del período)")

# Información sobre benchmark problemático
if st.session_state.get("benchmark", "ECH") == "^IPSA":
    st.warning("⚠️ **^IPSA tiene datos limitados en Yahoo Finance.** Considera usar **ECH** (Chile ETF) o **^GSPC** (S&P 500)")

st.divider()

# ===== BUSCADOR =====
st.subheader("🔎 Buscador de tickers")
search_col1, search_col2 = st.columns([2, 1])
with search_col1:
    query = st.text_input(
        "Buscar activos", 
        "",
        placeholder="Ej: 'santander', 'SQM', 'ENEL', 'SPY', 'AAPL'",
        help="Busca en la base local de acciones chilenas, ETFs y acciones globales"
    )
with search_col2:
    want_validate = st.toggle(
        "Validar datos", 
        value=True,
        help="Verifica que Yahoo Finance tenga datos disponibles"
    )

results = search_local_catalog(query) if query else []
if results:
    st.write("**Resultados encontrados:**")
    rs_df = pd.DataFrame(results, columns=["Ticker", "Descripción"])
    st.dataframe(rs_df, hide_index=True, use_container_width=True)

st.divider()

# ===== PORTAFOLIOS =====
st.subheader("📊 Gestión de Portafolios")

# Inicializar portafolios si no existen
if "portfolios" not in st.session_state:
    st.session_state.portfolios = []

# Botones de gestión
mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
with mgmt_col1:
    if st.button("➕ Agregar portafolio"):
        st.session_state.portfolios.append({
            "name": f"Cartera {len(st.session_state.portfolios) + 1}",
            "text": "# Formato: TICKER, peso\nSPY, 60\nVTI, 40\n"
        })
        st.rerun()

with mgmt_col2:
    if st.button("📋 Ejemplo demo") and len(st.session_state.portfolios) < 6:
        st.session_state.portfolios.append({
            "name": "Ejemplo: 60/40 US",
            "text": "SPY, 60\nBND, 40\n"
        })
        st.rerun()

with mgmt_col3:
    if st.button("🗑️ Limpiar todo") and st.session_state.portfolios:
        st.session_state.portfolios = []
        st.rerun()

# Mostrar portafolios
all_series = []
for idx, portfolio in enumerate(st.session_state.portfolios):
    with st.expander(f"📁 {portfolio['name']}", expanded=True):
        
        # Editar nombre y contenido
        name_col, del_col = st.columns([4, 1])
        with name_col:
            new_name = st.text_input(f"Nombre", portfolio["name"], key=f"name_{idx}")
            st.session_state.portfolios[idx]["name"] = new_name
        with del_col:
            if st.button("🗑️", key=f"del_{idx}", help="Eliminar este portafolio"):
                st.session_state.portfolios.pop(idx)
                st.rerun()

        new_text = st.text_area(
            f"Composición",
            portfolio["text"],
            height=150,
            key=f"text_{idx}",
            help="Formato: TICKER, peso (uno por línea). Ejemplo:\nSPY, 60\nVTI, 40"
        )
        st.session_state.portfolios[idx]["text"] = new_text

        # Procesar portafolio
        tickers, weights = parse_lines_to_weights(new_text)
        if not tickers:
            st.warning("⚠️ No se encontraron tickers válidos")
            continue

        # Descargar datos
        prices = load_prices(
            tickers,
            period=(None if use_dates else rango),
            interval=intervalo,
            start=(start_str if use_dates else None),
            end=(end_str if use_dates else None),
            fill_gaps=fill_gaps
        )

        if prices.empty:
            st.error("❌ No se pudieron obtener datos para estos tickers")
            continue

        # Construir portafolio
        port_series = build_portfolio_value(prices, tickers, weights, fill_gaps)
        if port_series.empty:
            st.error("❌ No se pudo construir el portafolio")
            continue

        # Métricas
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
            fv = b.first_valid_index()
            if fv is not None:
                bench_series = (b / b.loc[fv]) * 100.0

        # KPIs
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("**Retorno**", f"{met['ret']*100:,.1f}%", help="Ganancia total del período")
        kpi2.metric("**Volatilidad**", f"{met['vol']*100:,.1f}%", help="Riesgo anualizado (desviación estándar)")
        kpi3.metric("**Drawdown máx**", f"{met['dd']*100:,.1f}%", help="Peor caída desde un máximo")
        kpi4.metric("**Sharpe**", f"{met['sharpe']:.2f}", help="Retorno ajustado por riesgo")
        kpi5.metric("**CAGR**", f"{met['cagr']*100:,.1f}%", help="Tasa de crecimiento anual compuesta")

        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=port_series.index, 
            y=port_series.values,
            name=f"{new_name} (base=100)", 
            mode="lines",
            line=dict(width=2)
        ))
        
        if bench_series is not None:
            fig.add_trace(go.Scatter(
                x=bench_series.index, 
                y=bench_series.values,
                name=f"Benchmark {benchmark}", 
                mode="lines",
                line=dict(width=1, dash="dash")
            ))

        fig.update_layout(
            template="plotly_dark", 
            height=450,
            xaxis=dict(showgrid=True, title="Tiempo"),
            yaxis=dict(
                type="log" if usar_log else "linear", 
                title="Índice (base=100)"
            ),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de composición
        comp_df = pd.DataFrame({
            "Ticker": tickers, 
            "Peso": [f"{w:.1%}" for w in weights]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        all_series.append((new_name, port_series))

# ===== COMPARADOR GLOBAL =====
if len(all_series) > 1:
    st.divider()
    st.subheader("🔬 Comparador global")
    
    fig_all = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (name, series) in enumerate(all_series):
        fig_all.add_trace(go.Scatter(
            x=series.index, 
            y=series.values, 
            name=name, 
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    # Agregar benchmark al comparador
    if benchmark and not bench_prices.empty:
        b = bench_prices.iloc[:,0]
        fv = b.first_valid_index()
        if fv is not None:
            bser = (b / b.loc[fv]) * 100.0
            fig_all.add_trace(go.Scatter(
                x=bser.index, 
                y=bser.values, 
                name=f"Benchmark {benchmark}", 
                mode="lines",
                line=dict(color='gray', width=1, dash='dot')
            ))

    fig_all.update_layout(
        template="plotly_dark", 
        height=500,
        xaxis=dict(showgrid=True, title="Tiempo"),
        yaxis=dict(
            type="log" if usar_log else "linear", 
            title="Índice (base=100)"
        ),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_all, use_container_width=True)

# ===== Auto-actualizar URL =====
if not st.session_state.get("_restore_guard", False) and st.session_state.get("_initialized", False):
    try:
        current_snapshot = _get_state_snapshot()
        new_b64 = _encode_snapshot_to_url(current_snapshot)
        current_b64 = st.query_params.get("s", "")
        if new_b64 != current_b64:
            st.query_params["s"] = new_b64
    except Exception:
        pass