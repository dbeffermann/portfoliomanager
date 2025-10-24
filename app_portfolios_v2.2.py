import math, numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st, yfinance as yf
import json, base64

st.set_page_config(page_title="📈 Portafolios (multi) + Buscador", layout="wide")

# ===== Restauración inicial desde la URL (si hay ?s=...) =====
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
        st.session_state._restore_guard = True  # evita loops al setear query params
        
        # Solo aplicar si los valores son diferentes (evita resets innecesarios)
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
            
        # Fechas - solo si use_dates está activo
        if snap.get("use_dates", False):
            if snap.get("start") and st.session_state.get("start_str") != snap["start"]:
                st.session_state["start_str"] = snap["start"]
            if snap.get("end") and st.session_state.get("end_str") != snap["end"]:
                st.session_state["end_str"] = snap["end"]
                
        # Portafolios - solo si hay cambios significativos
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

# ---------------- Mini base de tickers CL (IPSA-ish) ----------------
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

def _get_state_snapshot():
    """Snapshot mínimo y portable del estado del usuario."""
    return {
        "rango": st.session_state.get("rango", "1y"),
        "intervalo": st.session_state.get("intervalo_raw", "1d"),
        "benchmark": st.session_state.get("benchmark", "^IPSA"),
        "use_dates": st.session_state.get("use_dates", False),
        "start": st.session_state.get("start_str", "") or "",
        "end": st.session_state.get("end_str", "") or "",
        "fill_gaps": st.session_state.get("fill_gaps", True),
        "portfolios": [
            {"name": p["name"], "text": p["text"]}
            for p in st.session_state.get("portfolios", [])
        ],
    }

# ----------------- Sidebar: Persistencia -----------------
with st.sidebar:
    st.markdown("## 💾 Guardar / Compartir")
    if st.button("🔗 Guardar estado en el link (URL)"):
        b64 = _encode_snapshot_to_url(_get_state_snapshot())
        st.query_params["s"] = b64
        st.success("Listo. El link actualizado quedó en la barra del navegador.")
        st.markdown(f"[Abrir este estado](?s={b64})")
    
    # Botón para resetear la aplicación
    if st.button("🔄 Resetear app (valores iniciales)", type="secondary"):
        # Limpiar la URL removiendo todos los parámetros
        st.query_params.clear()
        # Limpiar el session state (mantener solo las guardas necesarias)
        keys_to_remove = [k for k in st.session_state.keys() if k not in ["_restore_guard", "_initialized"]]
        for key in keys_to_remove:
            del st.session_state[key]
        # Resetear la bandera de inicialización para permitir nueva carga
        st.session_state._initialized = False
        st.success("App reseteada a valores iniciales.")
        st.rerun()

    # Exportar JSON
    snap_now = _get_state_snapshot()
    st.download_button(
        "⬇️ Exportar estado (JSON)",
        data=json.dumps(snap_now, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="portfolios_state.json",
        mime="application/json"
    )

    # Importar JSON
    up = st.file_uploader("⬆️ Importar estado (JSON)", type=["json"])
    if up is not None:
        try:
            imported = json.loads(up.read().decode("utf-8"))
            # Limpiar query params antes de importar para evitar conflictos
            st.query_params.clear()
            _apply_state_snapshot(imported)
            st.session_state._initialized = True
            st.success("Estado importado. Ya puedes seguir donde quedaste.")
            st.rerun()
        except Exception:
            st.error("Archivo inválido. Sube un JSON exportado desde la app.")

# ----------------- UI global (controles) -----------------
st.markdown("<h2 style='margin:0'>📈 Portafolios (múltiples) + Buscador de Tickers</h2>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2,1,1,1])
with c1:
    # Determine the index based on current session state value
    rango_options = ["3mo","6mo","1y","2y","5y","10y","max"]
    current_rango = st.session_state.get("rango", "1y")
    rango_index = rango_options.index(current_rango) if current_rango in rango_options else 2
    rango = st.selectbox("Rango", rango_options, index=rango_index, key="rango")
with c2:
    intervalo_raw = st.selectbox("Intervalo", ["1d","1wk","1mo"], index=0, key="intervalo_raw")
with c3:
    benchmark = st.text_input("Benchmark (p.ej. ^IPSA, ^GSPC)", st.session_state.get("benchmark", "^IPSA"), key="benchmark")
with c4:
    usar_log = st.toggle("Escala log", value=False, key="usar_log")

# Opciones adicionales
opt1, opt2 = st.columns(2)
with opt1:
    use_dates = st.toggle("Usar fechas exactas (start/end)", value=st.session_state.get("use_dates", False), key="use_dates")
with opt2:
    fill_gaps = st.toggle("Rellenar huecos (ffill)", value=st.session_state.get("fill_gaps", True), key="fill_gaps")

if use_dates:
    d1, d2 = st.columns(2)
    default_start = pd.to_datetime(st.session_state.get("start_str", "2018-01-01"))
    default_end   = pd.to_datetime(st.session_state.get("end_str", pd.Timestamp.today().strftime("%Y-%m-%d")))
    start_date = d1.date_input("Inicio", default_start)
    end_date   = d2.date_input("Fin",    default_end)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
    st.session_state["start_str"] = start_str
    st.session_state["end_str"]   = end_str
else:
    start_str = end_str = None

intervalo = choose_interval(rango, intervalo_raw)

st.caption("💡 Los gráficos muestran **índice base 100** (rendimiento relativo), no montos en CLP o USD.")
st.divider()

# ============================================================
# 🔀 TABS: Aprende / Guía técnica / Portafolios
# ============================================================
tab_aprende, tab_guia, tab_portafolios = st.tabs(["📘 Aprende", "🧩 Guía técnica de uso", "📊 Portafolios"])

# ============================================================
# 📘 TAB 1: Contenido educativo + ejemplo AAPL vs ^GSPC
# ============================================================
with tab_aprende:
    st.markdown("## 📘 Aprende a usar la app y entender tus resultados")
    st.info("Esta herramienta te permite explorar cómo evolucionan tus inversiones. "
            "Los gráficos muestran **índices base 100** (crecimiento porcentual relativo), "
            "no montos en pesos ni en dólares.")

    st.markdown("""
---
### 🎨 Personaliza tu experiencia
- **Modo oscuro/claro**: Haz clic en los **⋮** (tres puntos) en la esquina superior derecha → **Settings** → **Choose app theme** → **Dark/Light**
- **Pantalla completa**: Los gráficos cuentan con la opción **Fullscreen** para maximizar su tamaño
- **Ocultar sidebar**: Haz clic en la **←** para dar más espacio a los gráficos

---
### 💾 Guarda y comparte tu trabajo
**En el sidebar izquierdo encontrarás:**
- **🔗 Guardar estado en el link**: Actualiza la URL con toda tu configuración actual. Guarda el link como bookmark o compártelo.
- **� Resetear app**: Vuelve a los valores iniciales limpiando la URL y toda la configuración automáticamente.
- **�📥 Exportar JSON**: Descarga un archivo con todos tus portafolios y configuración.
- **📤 Importar JSON**: Carga una configuración previamente guardada.

💡 *Tip: El enlace se actualiza automáticamente mientras trabajas. Usa "Resetear app" para empezar limpio en cualquier momento.*    


                ---
### 1️⃣ Explora los datos sin miedo
- Cambia **rango** (3m, 6m, 1y, 5y, max).
- Ajusta **intervalo** (`1d`, `1wk`, `1mo`).
- Usa **fechas personalizadas** para períodos específicos.

💡 *Si la curva se suaviza al usar intervalos semanales, el activo es volátil día a día.*

---
### 2️⃣ Construye y compara carteras
- Crea combinaciones de activos con pesos distintos.
- Observa desempeño y estabilidad.
- Usa el comparador global para ver cuál resiste mejor las caídas.
- **Duplica portafolios** para crear variaciones rápidamente.

💡 *Curva más plana en crisis = mejor diversificación (menor drawdown).*

---
### 3️⃣ Analiza las métricas
| Métrica | Qué mide | Qué significa |
|:--|:--|:--|
| Retorno | Ganancia total | Crecimiento acumulado |
| Vol anual | Volatilidad | Qué tan errática es la curva |
| DD máx | Pérdida máxima | La peor caída desde un pico |
| Sharpe (≈) | Eficiencia riesgo/retorno | Mayor = más eficiente |
| CAGR (≈) | Crecimiento anual compuesto | Ritmo medio de avance |

---
### 4️⃣ Benchmark y comparación
- Usa **^IPSA** (Chile) o **^GSPC** (S&P 500) para comparar.
- Si tu cartera rinde más con menor volatilidad, vas ganando al mercado.
- El gráfico comparativo te muestra todos los portafolios juntos.

---
### 5️⃣ Buscador de activos
- Busca por nombre o símbolo: "ENEL", "SANTANDER", "SPY", "VOO"…
- Activa *Validar con Yahoo* para confirmar datos disponibles.
- La base de datos incluye acciones chilenas (`.SN`) y globales.

---
### 🔧 Funciones avanzadas
- **Rellenar vacíos**: Completa datos faltantes automáticamente.
- **Comparación global**: Ve todos tus portafolios en un solo gráfico.
- **Export/Import**: Respalda y comparte configuraciones completas.
- **URLs persistentes**: Cada cambio se guarda automáticamente en la URL.

---
### 💡 Consejos de uso
- **Experimenta sin temor**: Todos los cambios son temporales hasta que los guardes.
- **Usa rangos largos**: Para análisis de largo plazo, usa 5y o max.
- **Combina intervalos**: 1d para análisis detallado, 1wk para tendencias generales.
- **Compara siempre**: Usa benchmarks para contextualizar tus resultados.
""")

    # --- Ejemplo guiado ---
    st.markdown("---")
    st.markdown("### 🎯 Ejemplo guiado: Apple vs S&P 500 (2 años)")

    demo_period = "2y"
    demo_interval = "1wk"
    demo_tickers = ["AAPL", "^GSPC"]

    demo_prices = load_prices(
        demo_tickers,
        period=demo_period,
        interval=demo_interval,
        start=None,
        end=None,
        fill_gaps=True
    )

    if demo_prices.empty or any(col not in demo_prices.columns for col in demo_tickers):
        st.warning("No pude traer el ejemplo ahora. Vuelve a intentar más tarde.")
    else:
        def base100(s: pd.Series) -> pd.Series:
            fv = s.first_valid_index()
            if fv is None:
                return pd.Series(dtype=float)
            return (s / s.loc[fv]) * 100.0

        aapl_raw = demo_prices["AAPL"].dropna()
        spx_raw  = demo_prices["^GSPC"].dropna()

        aapl_idx = base100(aapl_raw)
        spx_idx  = base100(spx_raw)

        aapl_met = metrics_from_series(aapl_raw)
        spx_met  = metrics_from_series(spx_raw)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AAPL Retorno", f"{aapl_met['ret']*100:,.1f}%")
        c2.metric("AAPL Vol anual", f"{aapl_met['vol']*100:,.1f}%")
        c3.metric("AAPL DD máx", f"{aapl_met['dd']*100:,.1f}%")
        c4.metric("AAPL Sharpe (≈)", f"{aapl_met['sharpe']:.2f}")
        c5.metric("AAPL CAGR (≈)", f"{aapl_met['cagr']*100:,.1f}%")

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("S&P 500 Retorno", f"{spx_met['ret']*100:,.1f}%")
        d2.metric("S&P 500 Vol anual", f"{spx_met['vol']*100:,.1f}%")
        d3.metric("S&P 500 DD máx", f"{spx_met['dd']*100:,.1f}%")
        d4.metric("S&P 500 Sharpe (≈)", f"{spx_met['sharpe']:.2f}")
        d5.metric("S&P 500 CAGR (≈)", f"{spx_met['cagr']*100:,.1f}%")

        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(x=aapl_idx.index, y=aapl_idx.values, name="AAPL (base=100)", mode="lines"))
        fig_demo.add_trace(go.Scatter(x=spx_idx.index, y=spx_idx.values,  name="S&P 500 (base=100)", mode="lines"))
        fig_demo.update_layout(
            template="plotly_dark", height=480,
            xaxis=dict(showgrid=True, title="Tiempo"),
            yaxis=dict(title="Índice (base=100)"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_demo, use_container_width=True)

        def as_pct(x):
            return "—" if (x is None or pd.isna(x)) else f"{x*100:,.1f}%"

        st.markdown("#### 🧠 ¿Cómo interpretar este gráfico y las métricas?")
        bullets = []
        if pd.notna(aapl_met["ret"]) and pd.notna(spx_met["ret"]):
            if aapl_met["ret"] > spx_met["ret"]:
                bullets.append(f"- **Retorno:** AAPL supera al S&P 500 en el período ({as_pct(aapl_met['ret'])} vs {as_pct(spx_met['ret'])}).")
            else:
                bullets.append(f"- **Retorno:** AAPL rinde por **debajo** del S&P 500 ({as_pct(aapl_met['ret'])} vs {as_pct(spx_met['ret'])}).")
        if pd.notna(aapl_met["vol"]) and pd.notna(spx_met["vol"]):
            if aapl_met["vol"] > spx_met["vol"]:
                bullets.append(f"- **Riesgo/Volatilidad:** AAPL muestra **más volatilidad** ({as_pct(aapl_met['vol'])} vs {as_pct(spx_met['vol'])}).")
            else:
                bullets.append(f"- **Riesgo/Volatilidad:** AAPL es **más estable** ({as_pct(aapl_met['vol'])} vs {as_pct(spx_met['vol'])}).")
        if pd.notna(aapl_met["dd"]) and pd.notna(spx_met["dd"]):
            if aapl_met["dd"] < spx_met["dd"]:
                bullets.append(f"- **Caída máxima (DD):** AAPL sufrió **caídas más profundas** ({as_pct(aapl_met['dd'])} vs {as_pct(spx_met['dd'])}).")
            else:
                bullets.append(f"- **Caída máxima (DD):** AAPL **resistió mejor** ({as_pct(aapl_met['dd'])} vs {as_pct(spx_met['dd'])}).")
        if pd.notna(aapl_met["sharpe"]) and pd.notna(spx_met["sharpe"]):
            if aapl_met["sharpe"] > spx_met["sharpe"]:
                bullets.append(f"- **Eficiencia (Sharpe):** AAPL tiene **mejor relación retorno/riesgo**.")
            else:
                bullets.append(f"- **Eficiencia (Sharpe):** El índice tiene **mejor relación retorno/riesgo**.")
        if pd.notna(aapl_met["cagr"]) and pd.notna(spx_met["cagr"]):
            if aapl_met["cagr"] > spx_met["cagr"]:
                bullets.append(f"- **Crecimiento anual (CAGR):** AAPL crece a un ritmo anual **mayor**.")
            else:
                bullets.append(f"- **Crecimiento anual (CAGR):** El índice crece a un ritmo **mayor** que AAPL.")

        if bullets:
            st.markdown("\n".join(bullets))
        else:
            st.info("Revisa la conexión o intenta nuevamente: no se pudo generar la interpretación.")

        st.caption("🔎 Todo se muestra en **índice base 100**: compara rendimientos relativos sin depender de monedas (CLP/USD).")

# ============================================================
# 🧭 TAB 2: Guía práctica + carteras demo
# ============================================================
with tab_guia:
    st.subheader("🧩 Guía técnica de uso de la herramienta")
    st.info(
        "Esta sección explica cómo aprovechar las funciones analíticas de la app. "
        "No constituye asesoría financiera ni recomendación de inversión."
    )

    st.markdown("""
### 1) Objetivo general
Esta herramienta permite **analizar, comparar y simular carteras de inversión** usando datos reales de mercado.
Su propósito es educativo y analítico: entender cómo se comportan los portafolios ante diferentes combinaciones de activos.

### 2) Flujo de trabajo recomendado
1. **Buscar tickers:** usa el buscador para obtener datos de ETFs, acciones o índices (ej. `VOO`, `^GSPC`, `^IPSA`).
2. **Crear carteras:** combina varios activos asignando pesos (%) personalizados.
3. **Comparar carteras:** analiza rentabilidad, riesgo y correlación frente a benchmarks o entre sí.
4. **Guardar y restaurar:** exporta tu sesión como JSON o genera un link para continuar luego.

### 3) Funciones principales
- **Descarga automática:** los precios se obtienen desde Yahoo Finance en tiempo real (ajustados por splits y dividendos).
- **Visualización:** gráficos interactivos de rendimiento, drawdown y composición del portafolio.
- **Indicadores:** métricas como retorno anualizado, volatilidad, Sharpe ratio y correlaciones.
- **Persistencia:** la app guarda el estado en la URL o en un snapshot JSON.

### 4) Buenas prácticas de uso
- Verifica los tickers (algunos mercados usan sufijos como `.SN` o `.MX`).
- Mantén pesos que sumen 100% para resultados coherentes.
- Repite los análisis en distintos horizontes temporales (1, 3, 5 años) para evaluar consistencia.
- Actualiza manualmente si cambian tus datos base.

### 5) Próximas funcionalidades (roadmap)
- Integración con APIs locales (p. ej. BICE, Renta4, BTG).  
- Reportes PDF automáticos.  
- Módulo de optimización de portafolios (mínima varianza y frontera eficiente).  
""")


    st.markdown("---")
    st.markdown("### 🚀 Crear carteras demo (1 clic)")
    st.caption("Se crean tres carteras ejemplo con pesos razonables para empezar. Puedes editarlas después.")

    if st.button("Crear 3 carteras demo"):
        demo_portfolios = [
            {"name": "Conservadora", "text": "VOO, 0.60\n^IPSA, 0.40\n"},
            {"name": "Balanceada",   "text": "VOO, 0.60\nQQQ, 0.20\n^IPSA, 0.20\n"},
            {"name": "Agresiva",     "text": "QQQ, 0.50\nVOO, 0.20\nXLK, 0.15\nXLE, 0.10\nXLV, 0.05\n"},
        ]
        st.session_state.portfolios = demo_portfolios
        st.success("✅ Listo: se crearon las 3 carteras demo. Ve a la pestaña **📊 Portafolios** para verlas y editarlas.")
        st.rerun()

    if st.button("Limpiar mis carteras"):
        st.session_state.portfolios = [
            {"name": "Cartera 1", "text": "# TICKER, peso\n"},
            {"name": "Cartera 2", "text": "# TICKER, peso\n"},
            {"name": "Cartera 3", "text": "# TICKER, peso\n"},
        ]
        st.info("🧹 Se limpiaron las carteras. Pasa a **📊 Portafolios** para definirlas.")
        st.rerun()

# ============================================================
# 📊 TAB 3: Toda la lógica de portafolios
# ============================================================
with tab_portafolios:
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
    st.subheader("🧺 Define múltiples carteras")

    # Presets iniciales (editables)
    default_portfolios = {
    "Bancos CL": "BSANTANDER.SN, 0.35\nCHILE.SN, 0.35\nITAUCL.SN, 0.30\n",
    "Energía CL": "ENELAM.SN, 0.25\nENELCHILE.SN, 0.25\nCOLBUN.SN, 0.25\nAESGENER.SN, 0.25\n",
    "Retail CL": "CENCOSUD.SN, 0.25\nFALABELLA.SN, 0.25\nRIPLEY.SN, 0.25\nPARAUCO.SN, 0.25\n",
    "Infraestructura CL": "SALFACORP.SN, 0.25\nSOCOVESA.SN, 0.25\nCINTAC.SN, 0.25\nMADECO.SN, 0.25\n",
    "Mineras CL": "SQM-B.SN, 0.40\nCAP.SN, 0.30\nANTAISA.SN, 0.15\nIAM.SN, 0.15\n",
    "Telecom y Transporte CL": "ENTEL.SN, 0.40\nLATAM.SN, 0.30\nCCU.SN, 0.20\nNAVARINO.SN, 0.10\n",
    "IPSA ejemplo": "FALABELLA.SN, 0.15\nSQM-B.SN, 0.15\nCOPEC.SN, 0.15\nENELAM.SN, 0.15\nCAP.SN, 0.10\nCENCOSUD.SN, 0.10\nBCI.SN, 0.10\nPARAUCO.SN, 0.10\n",
    "USA Tech": "AAPL, 0.30\nMSFT, 0.30\nNVDA, 0.20\nGOOGL, 0.20\n"
}


    n = st.number_input("¿Cuántas carteras quieres analizar a la vez?", min_value=1, max_value=6, value=3, step=1)

    if "portfolios" not in st.session_state or not st.session_state.portfolios:
        st.session_state.portfolios = []
        for name, txt in list(default_portfolios.items())[:6]:
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
                period=(None if st.session_state.get("use_dates", False) else st.session_state.get("rango", "1y")),
                interval=choose_interval(st.session_state.get("rango", "1y"), st.session_state.get("intervalo_raw", "1d")),
                start=(st.session_state.get("start_str") if st.session_state.get("use_dates", False) else None),
                end=(st.session_state.get("end_str") if st.session_state.get("use_dates", False) else None),
                fill_gaps=st.session_state.get("fill_gaps", True)
            )

            if prices.empty:
                st.error("No hay datos. Revisa símbolos o rango/intervalo (prueba semanal '1wk' para ≥2y).")
                continue

            port_series = build_portfolio_value(prices, tickers, weights, fill_gaps=st.session_state.get("fill_gaps", True))
            if port_series.empty:
                st.error("No fue posible construir la serie del portafolio (sin datos válidos tras alinear).")
                continue

            met = metrics_from_series(port_series)

            # Benchmark
            bench_prices = load_prices(
                [st.session_state.get("benchmark", "^IPSA")],
                period=(None if st.session_state.get("use_dates", False) else st.session_state.get("rango", "1y")),
                interval=choose_interval(st.session_state.get("rango", "1y"), st.session_state.get("intervalo_raw", "1d")),
                start=(st.session_state.get("start_str") if st.session_state.get("use_dates", False) else None),
                end=(st.session_state.get("end_str") if st.session_state.get("use_dates", False) else None),
                fill_gaps=st.session_state.get("fill_gaps", True)
            )
            bench_series = None
            if not bench_prices.empty:
                b = bench_prices.iloc[:,0]
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
                                         name=f"Benchmark {st.session_state.get('benchmark', '^IPSA')}", mode="lines"))
            fig.update_layout(
                template="plotly_dark", height=520,
                xaxis=dict(showgrid=True),
                yaxis=dict(type="log" if st.session_state.get("usar_log", False) else "linear", title="Índice (base=100)"),
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
            [st.session_state.get("benchmark", "^IPSA")],
            period=(None if st.session_state.get("use_dates", False) else st.session_state.get("rango", "1y")),
            interval=choose_interval(st.session_state.get("rango", "1y"), st.session_state.get("intervalo_raw", "1d")),
            start=(st.session_state.get("start_str") if st.session_state.get("use_dates", False) else None),
            end=(st.session_state.get("end_str") if st.session_state.get("use_dates", False) else None),
            fill_gaps=st.session_state.get("fill_gaps", True)
        )
        if not bp.empty:
            bser = bp.iloc[:,0]
            fv = bser.first_valid_index()
            if fv is not None:
                bser = (bser / bser.loc[fv]) * 100.0
                fig_all.add_trace(go.Scatter(x=bser.index, y=bser.values, name=f"Benchmark {st.session_state.get('benchmark','^IPSA')}", mode="lines"))

        fig_all.update_layout(
            template="plotly_dark", height=520,
            xaxis=dict(showgrid=True),
            yaxis=dict(type="log" if st.session_state.get("usar_log", False) else "linear", title="Índice (base=100)"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=10,r=10,t=30,b=10)
        )
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.info("Agrega al menos una cartera para comparar.")

# ———— Auto-actualizar query param con el estado actual (una sola vez al final) ————
if not st.session_state.get("_restore_guard", False) and st.session_state.get("_initialized", False):
    try:
        current_snapshot = _get_state_snapshot()
        new_b64 = _encode_snapshot_to_url(current_snapshot)
        
        # Solo actualizar si realmente cambió algo significativo
        current_b64 = st.query_params.get("s", "")
        if new_b64 != current_b64:
            st.query_params["s"] = new_b64
    except Exception:
        pass
