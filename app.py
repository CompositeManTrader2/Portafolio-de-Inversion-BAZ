"""
Portafolio de Inversion BAZ - Punto Casa de Bolsa
Tablero de seguimiento diario de cartera institucional.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import base64
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from portfolio import analytics as an
from portfolio import engine as eng
from portfolio import loader as ld
from portfolio import market as mk
from portfolio import viz
from portfolio.taxonomy import (BENCHMARK_NOMBRE, BENCHMARK_TICKER, FX_TICKER,
                                CATALOGO)

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).parent
ASSETS = RAIZ / "assets"
DATOS = RAIZ / "data"

st.set_page_config(page_title="Portafolio de Inversion BAZ | Punto Casa de Bolsa",
                   page_icon="●", layout="wide",
                   initial_sidebar_state="expanded")

viz.registrar_plantilla()


# ==========================================================================
# Presentacion
# ==========================================================================

def cargar_estilos() -> None:
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def logo_svg(alto: int = 46) -> str:
    """
    Devuelve el logotipo como <img> con el SVG embebido en base64.

    No se inserta el SVG en linea porque el saneador de HTML de Streamlit
    descarta los nodos <text>, lo que dejaba el wordmark como texto suelto
    fuera de su posicion. Embebido como imagen, el vector llega intacto.
    """
    ruta = ASSETS / "logo.svg"
    if not ruta.exists():
        return ('<div style="font:600 0.9rem/1 Inter,sans-serif;color:#f4f3f7;'
                'letter-spacing:0.18em">PUNTO</div>')
    b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{b64}" '
            f'alt="Punto Casa de Bolsa" style="height:{alto}px;width:auto;'
            f'display:block" />')


def mxn(v: float, decimales: int = 0) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v:,.{decimales}f}"


def millones(v: float) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${v/1e6:,.2f} M"


def pct(v: float, decimales: int = 2, signo: bool = True) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.{decimales}f} %" if signo else f"{v:.{decimales}f} %"


def clase(v: float) -> str:
    if v is None or pd.isna(v):
        return "neutro"
    return "pos" if v > 0 else ("neg" if v < 0 else "neutro")


def tarjeta(etiqueta: str, valor: str, delta: str = "",
            clase_delta: str = "neutro", nota: str = "") -> str:
    partes = [
        '<div class="kpi">',
        f'<div class="kpi-etiqueta">{etiqueta}</div>',
        f'<div class="kpi-valor">{valor}</div>',
    ]
    if delta:
        partes.append(f'<div class="kpi-delta {clase_delta}">{delta}</div>')
    if nota:
        partes.append(f'<div class="kpi-nota">{nota}</div>')
    partes.append("</div>")
    return "".join(partes)


def tira(tarjetas: list[str], columnas: int | None = None) -> None:
    n = columnas or len(tarjetas)
    st.markdown(
        f'<div class="tira-kpi" style="grid-template-columns:repeat({n},1fr)">'
        + "".join(tarjetas) + "</div>",
        unsafe_allow_html=True,
    )


def panel(titulo: str, extra: str = "") -> None:
    st.markdown(
        f'<div class="panel-titulo"><span>{titulo}</span>'
        f'<span style="font-weight:500;letter-spacing:0.02em;color:#7d7a90">{extra}</span></div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# Datos
# ==========================================================================

@st.cache_data(show_spinner=False)
def _leer_base(bytes_o_ruta, hoja: str, sello: str):
    return ld.leer_posicion_base(bytes_o_ruta, hoja)


def archivo_por_defecto() -> Path | None:
    """
    Elige el archivo de posicion dentro de data/. Se prioriza el que lleva
    'posicion' en el nombre para no confundirlo con las boletas del custodio,
    que tambien son .xlsx y ordenan antes alfabeticamente.
    """
    if not DATOS.exists():
        return None
    candidatos = [p for p in sorted(DATOS.glob("*.xlsx"))
                  if not p.name.startswith("~$")]
    if not candidatos:
        return None
    for p in candidatos:
        if "posici" in p.name.lower():
            return p
    return candidatos[0]


def boletas_por_defecto(posicion: Path | None = None) -> list[Path]:
    """
    Archivos de operaciones incluidos en data/: todo .xlsx que no sea el de
    la posicion base.

    No se filtra por nombre a proposito. El custodio entrega las boletas con
    nomenclatura inconsistente ('2026.07.17.Res.104351...' y
    '2026_07_21_Res_104351...' son el mismo formato), y un patron fijo dejaba
    fuera archivos validos en silencio. Cualquier hoja sin operaciones legibles
    simplemente aporta cero, asi que incluir de mas no hace dano.
    """
    if not DATOS.exists():
        return []
    return [p for p in sorted(DATOS.glob("*.xlsx"))
            if not p.name.startswith("~$") and p != posicion]


def inicializar_estado() -> None:
    if "manuales" not in st.session_state:
        st.session_state.manuales = pd.DataFrame(columns=ld.COLUMNAS_MOV)


def registrar_manual(fecha_op, operacion, emisora, titulos, precio) -> None:
    fila = pd.DataFrame([{
        "fecha": fecha_op, "operacion": operacion, "emisora": emisora,
        "titulos": float(titulos), "precio": float(precio),
        "comision": None, "iva": None, "importe_neto": None,
        "fuente": "captura manual",
    }])
    st.session_state.manuales = pd.concat(
        [st.session_state.manuales, fila], ignore_index=True)


# ==========================================================================
# Barra lateral
# ==========================================================================

def barra_lateral():
    st.sidebar.markdown(
        f'<div style="padding:0.3rem 0 0.9rem">{logo_svg(40)}</div>',
        unsafe_allow_html=True)

    st.sidebar.markdown("### Posicion base")
    subido = st.sidebar.file_uploader(
        "Archivo de posicion (.xlsx)", type=["xlsx", "xlsm"],
        help="La hoja seleccionada debe contener Emisora, Titulos y Precio Compra.")

    origen = subido if subido is not None else archivo_por_defecto()
    if origen is None:
        st.sidebar.error("Carga un archivo de posicion para comenzar.")
        st.stop()

    if subido is not None:
        contenido = subido.getvalue()
        sello = f"{subido.name}-{len(contenido)}"
        fuente_nombre = subido.name
        ruta_posicion = None
    else:
        contenido = origen.read_bytes()
        sello = f"{origen.name}-{origen.stat().st_mtime}"
        fuente_nombre = origen.name
        ruta_posicion = origen

    try:
        hojas = ld._abrir(contenido).sheet_names
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    hoja = st.sidebar.selectbox("Hoja de la posicion base", hojas, index=0)
    fecha_base = st.sidebar.date_input("Fecha de la posicion base",
                                       value=ld.FECHA_BASE_DEFAULT)

    st.sidebar.markdown("### Movimientos")
    usar_coste = st.sidebar.checkbox(
        "Leer bitacora del mismo archivo", value=True,
        help="Busca una hoja 'Coste', 'Movimientos' u 'Operaciones'.")
    hoja_mov = None
    if usar_coste:
        opciones = ["(automatico)"] + hojas
        elegida = st.sidebar.selectbox("Hoja de movimientos", opciones, index=0)
        hoja_mov = None if elegida == "(automatico)" else elegida

    boletas = st.sidebar.file_uploader(
        "Archivos de operaciones (.xlsx)", type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        help="Boletas del custodio (Res.NNNNNN), reportes de la mesa o "
             "bitacoras propias. Se pueden cargar varios a la vez. Se leen "
             "todas las hojas del archivo y basta con que tengan Fecha, "
             "Operacion, Emisora, Titulos y Precio. Las boletas aportan ademas "
             "comision e IVA reales. Todo se concilia contra lo ya cargado sin "
             "duplicar operaciones.")

    st.sidebar.markdown("### Efectivo y costos")
    efectivo_inicial = st.sidebar.number_input(
        "Efectivo inicial (MXN)", value=0.0, step=100_000.0, format="%.2f",
        help="Saldo de efectivo en la fecha de la posicion base. Si se deja "
             "en cero, el saldo mostrado es el flujo neto acumulado por las "
             "operaciones.")
    comision_bps = st.sidebar.number_input(
        "Comision (puntos base)", value=eng.COMISION_BPS_DEFAULT,
        min_value=0.0, max_value=100.0, step=0.5,
        help="Calibrado en 4 pb contra la boleta Res.104351.")
    tasa_iva = st.sidebar.number_input(
        "IVA sobre comision", value=eng.TASA_IVA_DEFAULT,
        min_value=0.0, max_value=0.30, step=0.01, format="%.2f")

    st.sidebar.markdown("### Analisis")
    ventana = st.sidebar.select_slider(
        "Ventana de historico (sesiones)",
        options=[60, 90, 120, 180, 252, 400], value=252)
    tasa_libre = st.sidebar.number_input(
        "Tasa libre de riesgo anual", value=0.09, min_value=0.0,
        max_value=0.25, step=0.005, format="%.3f",
        help="Referencia para Sharpe y Sortino. Cetes 28 dias.")

    if st.sidebar.button("Actualizar precios", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    return dict(contenido=contenido, sello=sello, hoja=hoja,
                fecha_base=fecha_base, usar_coste=usar_coste, hoja_mov=hoja_mov,
                boletas=boletas, efectivo_inicial=efectivo_inicial,
                comision_bps=comision_bps, tasa_iva=tasa_iva,
                ventana=ventana, tasa_libre=tasa_libre,
                fuente_nombre=fuente_nombre, ruta_posicion=ruta_posicion)


# ==========================================================================
# Pestanas
# ==========================================================================

def pestana_resumen(val, res, resumen, hist, bench_hist, riesgo, conc, fx):
    izq, der = st.columns([1.35, 1])

    with izq:
        panel("Composicion por sector", f"{len(val)} posiciones")
        st.plotly_chart(viz.treemap_composicion(val, "sector"),
                        width="stretch", config={"displayModeBar": False})
    with der:
        panel("Desempeno relativo", f"base 100 · {BENCHMARK_NOMBRE}")
        serie = an.serie_valor_portafolio(hist, val)
        st.plotly_chart(
            viz.linea_desempeno(serie, bench_hist, "IPC"),
            width="stretch", config={"displayModeBar": False})

    izq2, der2 = st.columns(2)
    with izq2:
        panel("Mayores aportes al resultado", "P&L no realizado")
        st.plotly_chart(viz.barras_contribucion(val, "emisora", "no_realizado"),
                        width="stretch", config={"displayModeBar": False})
    with der2:
        panel("Movimiento del dia", "variacion contra cierre previo")
        st.plotly_chart(
            viz.barras_contribucion(val, "emisora", "pnl_dia",
                                    "P&L del dia (MXN)"),
            width="stretch", config={"displayModeBar": False})

    panel("Indicadores de riesgo y concentracion")
    tira([
        tarjeta("Volatilidad anual", pct(riesgo["vol_anual"], 1, False)),
        tarjeta("Sharpe", f"{riesgo['sharpe']:.2f}" if riesgo["sharpe"] == riesgo["sharpe"] else "—"),
        tarjeta("Beta vs IPC", f"{riesgo['beta']:.2f}" if riesgo["beta"] == riesgo["beta"] else "—"),
        tarjeta("Caida maxima", pct(riesgo["max_drawdown"], 1)),
        tarjeta("VaR 95 % diario", pct(riesgo["var_95"], 2)),
        tarjeta("Posiciones efectivas",
                f"{conc['n_efectivo']:.1f}" if conc["n_efectivo"] == conc["n_efectivo"] else "—",
                nota=f"top 5: {conc['top5_pct']:.0f} %"),
        tarjeta("Exposicion USD", pct(fx["usd_pct"], 1, False),
                nota=f"{millones(fx['usd_monto'])}"),
    ], columnas=7)


def pestana_posiciones(val):
    panel("Detalle de posiciones", "valuacion a ultimo precio disponible")

    cols = ["emisora", "sector", "industria", "region", "mercado", "clase_activo",
            "titulos", "precio_costo", "precio_mercado", "var_dia_pct",
            "costo_total", "valor_mercado", "no_realizado", "rend_pct",
            "pnl_dia", "peso_pct"]
    d = val[[c for c in cols if c in val.columns]].copy()

    st.dataframe(
        d, width="stretch", height=620, hide_index=True,
        column_config={
            "emisora": st.column_config.TextColumn("Emisora", width="medium"),
            "sector": st.column_config.TextColumn("Sector"),
            "industria": st.column_config.TextColumn("Industria"),
            "region": st.column_config.TextColumn("Region"),
            "mercado": st.column_config.TextColumn("Mercado", width="small"),
            "clase_activo": st.column_config.TextColumn("Clase"),
            "titulos": st.column_config.NumberColumn("Titulos", format="localized"),
            "precio_costo": st.column_config.NumberColumn("P. costo", format="%.4f"),
            "precio_mercado": st.column_config.NumberColumn("P. mercado", format="%.4f"),
            "var_dia_pct": st.column_config.NumberColumn("Var. dia", format="%+.2f %%"),
            "costo_total": st.column_config.NumberColumn("Costo", format="localized"),
            "valor_mercado": st.column_config.NumberColumn("Valor mdo.", format="localized"),
            "no_realizado": st.column_config.NumberColumn("P&L", format="localized"),
            "rend_pct": st.column_config.NumberColumn("Rend.", format="%+.2f %%"),
            "pnl_dia": st.column_config.NumberColumn("P&L dia", format="localized"),
            "peso_pct": st.column_config.ProgressColumn(
                "Peso", format="%.2f %%", min_value=0.0,
                max_value=float(val["peso_pct"].max()) if len(val) else 1.0),
        },
    )

    if val["precio_estimado"].any():
        faltan = ", ".join(val.loc[val["precio_estimado"], "emisora"].astype(str))
        st.warning(
            f"Sin precio vivo para: {faltan}. Se valuaron con el precio de "
            f"referencia del archivo o, en su defecto, con el costo.")

    st.download_button(
        "Descargar posiciones (CSV)",
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"posiciones_baz_{date.today():%Y%m%d}.csv",
        mime="text/csv")


def pestana_segmentacion(val, efectivo):
    dimensiones = [("sector", "Sector"), ("region", "Region geografica"),
                   ("clase_activo", "Clase de activo"), ("mercado", "Mercado"),
                   ("industria", "Industria"), ("divisa_subyacente", "Divisa subyacente")]

    elegida = st.radio("Segmentar por", [d[1] for d in dimensiones],
                       horizontal=True, label_visibility="collapsed")
    dim = next(d[0] for d in dimensiones if d[1] == elegida)

    incluir_efectivo = st.checkbox("Incluir efectivo en la distribucion",
                                   value=False)
    g = eng.agrupar(val, dim, efectivo, incluir_efectivo)

    izq, der = st.columns([1, 1])
    with izq:
        panel(f"Distribucion por {elegida.lower()}")
        st.plotly_chart(viz.barras_dimension(g, dim),
                        width="stretch", config={"displayModeBar": False})
    with der:
        panel(f"Resultado por {elegida.lower()}")
        st.plotly_chart(
            viz.barras_contribucion(g, dim, "no_realizado", n=12),
            width="stretch", config={"displayModeBar": False})

    panel("Tabla de segmentacion")
    st.dataframe(
        g, width="stretch", hide_index=True,
        column_config={
            dim: st.column_config.TextColumn(elegida, width="medium"),
            "valor_mercado": st.column_config.NumberColumn("Valor mdo.", format="localized"),
            "costo_total": st.column_config.NumberColumn("Costo", format="localized"),
            "no_realizado": st.column_config.NumberColumn("P&L", format="localized"),
            "pnl_dia": st.column_config.NumberColumn("P&L dia", format="localized"),
            "n_posiciones": st.column_config.NumberColumn("Posiciones", format="%d"),
            "peso_pct": st.column_config.NumberColumn("Peso", format="%.2f %%"),
            "rend_pct": st.column_config.NumberColumn("Rend.", format="%+.2f %%"),
        })


def pestana_riesgo(val, hist, bench_hist, riesgo, conc, riesgo_pos, tecnicos,
                   sin_historico):
    panel("Metricas de riesgo", f"{riesgo['sesiones']} sesiones · anualizado 252d")

    if sin_historico:
        st.caption(
            f"Fuera del bloque de riesgo por historico insuficiente: "
            f"{', '.join(sin_historico)}. Estas posiciones se valuan con normalidad, "
            f"pero no entran en volatilidad, correlaciones ni contribucion al riesgo.")
    tira([
        tarjeta("Volatilidad", pct(riesgo["vol_anual"], 1, False)),
        tarjeta("Sharpe", f"{riesgo['sharpe']:.2f}" if riesgo["sharpe"] == riesgo["sharpe"] else "—"),
        tarjeta("Sortino", f"{riesgo['sortino']:.2f}" if riesgo["sortino"] == riesgo["sortino"] else "—"),
        tarjeta("Beta", f"{riesgo['beta']:.2f}" if riesgo["beta"] == riesgo["beta"] else "—"),
        tarjeta("Alfa anual", pct(riesgo["alpha_anual"], 1)),
        tarjeta("Tracking error", pct(riesgo["tracking_error"], 1, False)),
        tarjeta("Information ratio",
                f"{riesgo['info_ratio']:.2f}" if riesgo["info_ratio"] == riesgo["info_ratio"] else "—"),
    ], columnas=7)

    tira([
        tarjeta("VaR 95 % diario", pct(riesgo["var_95"], 2),
                nota="percentil historico"),
        tarjeta("VaR 99 % diario", pct(riesgo["var_99"], 2)),
        tarjeta("CVaR 95 %", pct(riesgo["cvar_95"], 2),
                nota="perdida media en la cola"),
        tarjeta("Caida maxima", pct(riesgo["max_drawdown"], 1)),
        tarjeta("Herfindahl", f"{conc['hhi']:,.0f}",
                nota="escala 0 – 10,000"),
        tarjeta("Concentracion top 10", pct(conc["top10_pct"], 1, False)),
    ], columnas=6)

    izq, der = st.columns([1.15, 1])
    with izq:
        panel("Peso contra contribucion al riesgo",
              "una barra de riesgo mayor que la de peso indica sobreexposicion")
        st.plotly_chart(viz.barras_riesgo_vs_peso(riesgo_pos),
                        width="stretch", config={"displayModeBar": False})
    with der:
        panel("Riesgo contra rendimiento", "el tamano codifica el peso")
        st.plotly_chart(
            viz.dispersion_riesgo_rendimiento(val, tecnicos, riesgo_pos),
            width="stretch", config={"displayModeBar": False})

    panel("Caida acumulada desde maximo", "cartera vigente a titulos constantes")
    st.plotly_chart(viz.area_drawdown(an.serie_valor_portafolio(hist, val)),
                    width="stretch", config={"displayModeBar": False})

    panel("Correlacion entre las mayores posiciones",
          "correlaciones altas reducen la diversificacion efectiva")
    st.plotly_chart(viz.mapa_correlacion(an.matriz_correlacion(hist, val)),
                    width="stretch", config={"displayModeBar": False})


def pestana_atribucion(val, tecnicos):
    dim = st.radio("Atribuir por", ["sector", "region", "clase_activo", "industria"],
                   horizontal=True, format_func=str.capitalize,
                   label_visibility="collapsed")

    atr = an.atribucion(val, dim)
    panel(f"Contribucion al resultado por {dim}",
          "la suma de las contribuciones reproduce el rendimiento del portafolio")
    st.dataframe(
        atr, width="stretch", hide_index=True,
        column_config={
            dim: st.column_config.TextColumn(dim.capitalize(), width="medium"),
            "valor_mercado": st.column_config.NumberColumn("Valor mdo.", format="localized"),
            "costo_total": st.column_config.NumberColumn("Costo", format="localized"),
            "no_realizado": st.column_config.NumberColumn("P&L", format="localized"),
            "pnl_dia": st.column_config.NumberColumn("P&L dia", format="localized"),
            "n": st.column_config.NumberColumn("Pos.", format="%d"),
            "rend_pct": st.column_config.NumberColumn("Rend.", format="%+.2f %%"),
            "peso_pct": st.column_config.NumberColumn("Peso", format="%.2f %%"),
            "contrib_pct": st.column_config.NumberColumn("Contribucion", format="%+.3f %%"),
        })

    ganadores, perdedores = an.ganadores_perdedores(val, 8)
    izq, der = st.columns(2)
    cfg = {
        "emisora": st.column_config.TextColumn("Emisora", width="medium"),
        "sector": st.column_config.TextColumn("Sector"),
        "region": st.column_config.TextColumn("Region"),
        "titulos": st.column_config.NumberColumn("Titulos", format="localized"),
        "precio_costo": st.column_config.NumberColumn("P. costo", format="%.2f"),
        "precio_mercado": st.column_config.NumberColumn("P. mdo.", format="%.2f"),
        "valor_mercado": st.column_config.NumberColumn("Valor", format="localized"),
        "no_realizado": st.column_config.NumberColumn("P&L", format="localized"),
        "rend_pct": st.column_config.NumberColumn("Rend.", format="%+.2f %%"),
        "peso_pct": st.column_config.NumberColumn("Peso", format="%.2f %%"),
        "pnl_dia": st.column_config.NumberColumn("P&L dia", format="localized"),
    }
    with izq:
        panel("Lo que esta funcionando", "mayores aportes en pesos")
        st.dataframe(ganadores, width="stretch", hide_index=True, column_config=cfg)
    with der:
        panel("Lo que no esta funcionando", "mayores detracciones en pesos")
        st.dataframe(perdedores, width="stretch", hide_index=True, column_config=cfg)

    if tecnicos is not None and len(tecnicos):
        panel("Senales tecnicas", "medias moviles, RSI de 14 sesiones y momentum")
        t = tecnicos.merge(val[["ticker", "emisora", "sector", "peso_pct"]],
                           on="ticker", how="inner")
        cols = ["emisora", "sector", "peso_pct", "vs_media50_pct",
                "vs_media200_pct", "desde_max52_pct", "rsi14",
                "momentum_1m_pct", "momentum_3m_pct"]
        st.dataframe(
            t[cols].sort_values("momentum_3m_pct", ascending=False),
            width="stretch", hide_index=True, height=420,
            column_config={
                "emisora": st.column_config.TextColumn("Emisora", width="medium"),
                "sector": st.column_config.TextColumn("Sector"),
                "peso_pct": st.column_config.NumberColumn("Peso", format="%.2f %%"),
                "vs_media50_pct": st.column_config.NumberColumn("vs MM50", format="%+.1f %%"),
                "vs_media200_pct": st.column_config.NumberColumn("vs MM200", format="%+.1f %%"),
                "desde_max52_pct": st.column_config.NumberColumn("Desde max 52s", format="%+.1f %%"),
                "rsi14": st.column_config.NumberColumn("RSI 14", format="%.0f"),
                "momentum_1m_pct": st.column_config.NumberColumn("Mom. 1m", format="%+.1f %%"),
                "momentum_3m_pct": st.column_config.NumberColumn("Mom. 3m", format="%+.1f %%"),
            })


def pestana_efectivo(res, resumen, fx):
    panel("Esquema de efectivo",
          "el saldo se construye a partir de los flujos de compra y venta")
    tira([
        tarjeta("Efectivo inicial", millones(res.efectivo_inicial)),
        tarjeta("Ventas (bruto)", millones(res.flujo_ventas)),
        tarjeta("Compras (bruto)", millones(-res.flujo_compras)),
        tarjeta("Comisiones e IVA", millones(-res.costos_totales)),
        tarjeta("Saldo actual", millones(res.efectivo),
                clase_delta=clase(res.efectivo)),
        tarjeta("Peso en portafolio", pct(resumen["peso_efectivo_pct"], 2, False)),
        tarjeta("Resultado realizado", millones(res.realizado),
                clase_delta=clase(res.realizado)),
    ], columnas=7)

    if res.efectivo < 0:
        st.info(
            f"El saldo de efectivo es negativo ({millones(res.efectivo)}) porque "
            f"las compras del periodo superan a las ventas y el efectivo inicial "
            f"esta capturado en {millones(res.efectivo_inicial)}. Ajusta el campo "
            f"«Efectivo inicial» en el panel lateral con el saldo real del "
            f"contrato a la fecha de la posicion base para que el esquema cuadre.")

    serie = eng.serie_efectivo(res.bitacora, res.efectivo_inicial)
    izq, der = st.columns([1.25, 1])
    with izq:
        panel("Evolucion del saldo")
        st.plotly_chart(viz.escalones_efectivo(serie),
                        width="stretch", config={"displayModeBar": False})
    with der:
        panel("Flujo neto por dia")
        st.plotly_chart(viz.barras_flujo(res.bitacora),
                        width="stretch", config={"displayModeBar": False})

    panel("Exposicion cambiaria", "las emisoras del SIC cotizan en pesos "
                                  "pero su valor subyacente esta en dolares")
    tira([
        tarjeta("Valor en USD subyacente", millones(fx["usd_monto"]),
                nota=f"{fx['usd_pct']:.1f} % del portafolio"),
        tarjeta("Valor en MXN", millones(fx["mxn_monto"])),
        tarjeta("Impacto de 1 % en USDMXN", millones(fx["impacto_1pct_fx"]),
                nota="sensibilidad lineal"),
    ], columnas=3)


def pestana_operaciones(res, cfg):
    panel("Registrar operacion", "se aplica de inmediato sobre la posicion")

    with st.form("alta_operacion", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([1.1, 0.8, 1.5, 1, 1])
        with c1:
            f_op = st.date_input("Fecha", value=date.today())
        with c2:
            op = st.selectbox("Operacion", ["C", "V"],
                              format_func=lambda x: "Compra" if x == "C" else "Venta")
        with c3:
            emisora = st.selectbox("Emisora", sorted(CATALOGO.keys()))
        with c4:
            titulos = st.number_input("Titulos", min_value=0.0, step=1000.0,
                                      format="%.0f")
        with c5:
            precio = st.number_input("Precio", min_value=0.0, step=0.01,
                                     format="%.6f")

        if st.form_submit_button("Registrar operacion", width="stretch"):
            if titulos <= 0 or precio <= 0:
                st.error("Titulos y precio deben ser mayores que cero.")
            else:
                registrar_manual(f_op, op, emisora, titulos, precio)
                st.success(
                    f"{'Compra' if op == 'C' else 'Venta'} de {titulos:,.0f} "
                    f"titulos de {emisora} a {precio:,.4f} registrada.")
                st.rerun()

    if len(st.session_state.manuales):
        panel("Operaciones capturadas en esta sesion",
              "no se guardan al cerrar; expórtalas si quieres conservarlas")
        st.dataframe(st.session_state.manuales, width="stretch", hide_index=True)
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Limpiar capturadas"):
                st.session_state.manuales = pd.DataFrame(columns=ld.COLUMNAS_MOV)
                st.rerun()
        with c2:
            st.download_button(
                "Descargar capturadas (CSV)",
                st.session_state.manuales.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"operaciones_capturadas_{date.today():%Y%m%d}.csv",
                mime="text/csv")

    panel("Bitacora consolidada",
          f"{len(res.bitacora)} operaciones · comision {cfg['comision_bps']:.1f} pb "
          f"+ IVA {cfg['tasa_iva']:.0%}")

    if not len(res.bitacora):
        st.info("Aun no hay movimientos aplicados sobre la posicion base.")
        return

    b = res.bitacora.copy()
    st.dataframe(
        b, width="stretch", hide_index=True, height=440,
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD MMM YYYY"),
            "operacion": st.column_config.TextColumn("Op.", width="small"),
            "emisora": st.column_config.TextColumn("Emisora", width="medium"),
            "titulos": st.column_config.NumberColumn("Titulos", format="localized"),
            "precio": st.column_config.NumberColumn("Precio", format="%.6f"),
            "monto_bruto": st.column_config.NumberColumn("Bruto", format="localized"),
            "comision": st.column_config.NumberColumn("Comision", format="localized"),
            "iva": st.column_config.NumberColumn("IVA", format="localized"),
            "monto_neto": st.column_config.NumberColumn("Neto", format="localized"),
            "efecto_caja": st.column_config.NumberColumn("Efecto caja", format="localized"),
            "costo_promedio_previo": st.column_config.NumberColumn("Costo previo", format="%.6f"),
            "resultado_realizado": st.column_config.NumberColumn("Realizado", format="localized"),
            "titulos_despues": st.column_config.NumberColumn("Titulos post", format="localized"),
            "costo_promedio_despues": st.column_config.NumberColumn("Costo post", format="%.6f"),
            "efectivo_despues": st.column_config.NumberColumn("Efectivo post", format="localized"),
            "fuente": st.column_config.TextColumn("Fuente"),
        })

    st.download_button(
        "Descargar bitacora (CSV)",
        b.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bitacora_baz_{date.today():%Y%m%d}.csv", mime="text/csv")


def pestana_diagnostico(hallazgos, res):
    panel("Diagnostico automatico",
          "reglas de mesa aplicadas sobre las metricas del portafolio")

    if not hallazgos:
        st.success("No se detectaron desviaciones relevantes.")
    else:
        iconos = {"critico": "‼", "atencion": "▲", "favorable": "✓"}
        for h in hallazgos:
            st.markdown(
                f'<div class="hallazgo {h["severidad"]}">'
                f'<div class="hallazgo-icono">{iconos.get(h["severidad"], "•")}</div>'
                f'<div><div class="hallazgo-titulo">{h["titulo"]}</div>'
                f'<div class="hallazgo-detalle">{h["detalle"]}</div></div></div>',
                unsafe_allow_html=True)

    if res.incidencias:
        panel("Incidencias de conciliacion")
        for i in res.incidencias:
            st.warning(i)

    panel("Hoja de ruta del modelo", "mejoras sugeridas por orden de impacto")
    st.markdown(MEJORAS)


MEJORAS = """
**1. Rendimiento ponderado por tiempo (TWR) y por dinero (MWR).**
Hoy el resultado se mide contra el costo promedio, lo que mezcla el efecto de
las aportaciones con el de la gestion. Un TWR diario encadenado aisla la
habilidad del gestor y es lo que exige GIPS para presentar cifras a un cliente
institucional; el MWR (TIR) mide lo que efectivamente gano el cliente. La
diferencia entre ambos cuantifica si el *timing* de las entradas sumo o resto.

**2. Atribucion de Brinson-Fachler contra el IPC.**
Separar el resultado en efecto asignacion (pesar de mas o de menos un sector),
efecto seleccion (elegir bien dentro del sector) e interaccion. Es la respuesta
directa a *que esta funcionando*: permite decir si el valor viene de la vision
sectorial o de la seleccion de emisoras. Requiere los pesos del indice.

**3. Modelo de riesgo multifactor.**
Sustituir la beta unica por exposiciones a factores (valor, tamano, momentum,
calidad, y para Mexico: sensibilidad a tasas, al dolar y al ciclo de consumo).
Revelaria, por ejemplo, si la cartera tiene una apuesta involuntaria y
concentrada al consumo domestico mexicano, que hoy solo se intuye por el peso
sectorial.

**4. Limites de mandato con alertas.**
Parametrizar los limites del contrato (maximo por emisora, por sector, por
mercado, minimo de liquidez, VaR maximo) y marcar en el tablero cada
incumplimiento. Convierte el tablero de descriptivo a operativo.

**5. Backtest de las decisiones de la mesa.**
Comparar el portafolio real contra un contrafactual que no hubiera operado
desde la posicion base. Responde con evidencia si las operaciones agregaron o
destruyeron valor, que es distinto de si el portafolio subio.

**6. Costos de transaccion e *implementation shortfall*.**
Ya se capturan comision e IVA. El siguiente paso es medir la diferencia entre
el precio de decision y el precio de ejecucion para dimensionar el costo real
de operar, que en emisoras de baja bursatilidad mexicana puede superar a la
comision explicita.

**7. Persistencia historica.**
Guardar una foto diaria de la posicion en una base ligera (SQLite o Parquet)
para construir la serie real del portafolio. Hoy el historico se aproxima
aplicando la tenencia actual sobre precios pasados, lo que no captura el efecto
de las operaciones intermedias.

**8. Riesgo de liquidez.**
Cruzar el tamano de cada posicion contra el volumen promedio diario para
estimar en cuantas sesiones se podria deshacer sin mover el precio. Relevante
en nombres como TRAXION o LACOMER, de menor bursatilidad.

**9. Escenarios y pruebas de estres.**
Choques parametricos (IPC −10 %, USDMXN +10 %, tasas +100 pb) y escenarios
historicos (marzo 2020, choque electoral de junio 2024) aplicados sobre las
sensibilidades de la cartera.

**10. Ingresos por dividendos.**
Incorporar el calendario de dividendos y su efecto en el efectivo. En una
cartera con FEMSA, AC, WALMEX y FUNO el flujo de dividendos es material y hoy
no se refleja en el esquema de efectivo.
"""


# ==========================================================================
# Principal
# ==========================================================================

def main() -> None:
    cargar_estilos()
    inicializar_estado()
    cfg = barra_lateral()

    # --- Carga y consolidacion -------------------------------------------
    try:
        base = _leer_base(cfg["contenido"], cfg["hoja"], cfg["sello"])
    except ld.ErrorDeCarga as e:
        st.error(f"No fue posible leer la posicion base: {e}")
        st.stop()

    # Los archivos cargados van primero: aportan comision e IVA reales y, al
    # conciliar, esos valores prevalecen sobre la estimacion por tarifa.
    bloques = []
    cargados: list[tuple[str, int]] = []
    for archivo in (cfg["boletas"] or []):
        try:
            leidas = ld.leer_operaciones(archivo.getvalue(),
                                         fuente=f"archivo {archivo.name}")
            bloques.append(leidas)
            cargados.append((archivo.name, len(leidas)))
        except Exception as e:
            st.sidebar.error(f"No se pudo leer {archivo.name}: {e}")

    for ruta in boletas_por_defecto(cfg.get("ruta_posicion")):
        try:
            bloques.append(ld.leer_operaciones(
                ruta.read_bytes(), fuente=f"boleta {ruta.stem[:14]}"))
        except Exception:
            pass

    if cfg["usar_coste"]:
        try:
            bloques.append(ld.leer_movimientos(cfg["contenido"], cfg["hoja_mov"]))
        except Exception:
            pass

    if len(st.session_state.manuales):
        bloques.append(st.session_state.manuales)

    # Un archivo que no aporta nada se reporta: un cero silencioso se lee como
    # "ya quedo cargado" cuando en realidad no se leyo ninguna operacion.
    for nombre, n in cargados:
        if n:
            st.sidebar.success(f"{nombre}: {n} operaciones leidas.")
        else:
            st.sidebar.warning(
                f"{nombre}: no se encontraron operaciones. Revisa que alguna "
                f"hoja tenga las columnas Operacion, Emisora, Titulos y Precio.")

    movimientos = ld.consolidar_movimientos(*bloques)

    res = eng.construir_posicion(
        base.df, movimientos, cfg["efectivo_inicial"],
        eng.CostosOperacion(cfg["comision_bps"], cfg["tasa_iva"]))

    # --- Mercado ----------------------------------------------------------
    tickers = tuple(t for t in res.posiciones["ticker"] if t)
    with st.spinner("Consultando precios de mercado…"):
        vigentes = mk.precios_vigentes(tickers)
        historico = mk.descargar_historico(tickers + (BENCHMARK_TICKER,),
                                           dias=int(cfg["ventana"] * 1.5))
        fx_dato = mk.tipo_de_cambio(FX_TICKER)

    precios, previos = mk.mapas_de_precio(vigentes)
    val = eng.valuar(res.posiciones, precios, previos)
    resumen = eng.resumen_cartera(val, res.efectivo, res.realizado)

    hist_activos = (historico.drop(columns=[BENCHMARK_TICKER], errors="ignore")
                    if len(historico) else pd.DataFrame())
    bench_hist = (historico[BENCHMARK_TICKER].dropna()
                  if len(historico) and BENCHMARK_TICKER in historico.columns
                  else None)

    rend_port = an.rendimientos_portafolio(hist_activos, val)
    rend_bench = None
    if bench_hist is not None and len(bench_hist) > 1:
        rend_bench = np.log(bench_hist / bench_hist.shift(1)).dropna()

    riesgo = an.metricas_riesgo(rend_port, rend_bench, cfg["tasa_libre"])
    conc = an.metricas_concentracion(val)
    riesgo_pos = an.descomposicion_riesgo(hist_activos, val)
    tecnicos = mk.indicadores_tecnicos(hist_activos) if len(hist_activos) else pd.DataFrame()
    fx = an.exposicion_fx(val)
    _, sin_historico = an.cobertura_historica(hist_activos, val)
    hallazgos = an.diagnostico(val, riesgo, conc, tecnicos, riesgo_pos,
                               resumen["peso_efectivo_pct"])

    # --- Encabezado -------------------------------------------------------
    sello_precio = "—"
    if len(vigentes) and vigentes["fecha_precio"].notna().any():
        sello_precio = pd.to_datetime(
            vigentes["fecha_precio"].max()).strftime("%d %b %Y")

    st.markdown(
        f'<div class="cabecera">'
        f'<div class="cabecera-logo">{logo_svg()}</div>'
        f'<div style="text-align:right">'
        f'<div class="cabecera-titulo">Portafolio de Inversion BAZ</div>'
        f'<div class="cabecera-sub">'
        f'Posicion base {base.fecha:%d %b %Y} · hoja «{base.hoja}» · '
        f'Precios al {sello_precio} · '
        f'USDMXN {fx_dato["valor"]:,.4f} · '
        f'Consulta {datetime.now():%d %b %Y %H:%M}'
        f'</div></div></div>',
        unsafe_allow_html=True)

    # --- Indicadores de cabecera ------------------------------------------
    tira([
        tarjeta("Valor del portafolio", millones(resumen["valor_total"]),
                nota=f"{resumen['n_posiciones']} posiciones + efectivo"),
        tarjeta("Valor en instrumentos", millones(resumen["valor_instrumentos"])),
        tarjeta("Costo de adquisicion", millones(resumen["costo_total"])),
        tarjeta("P&L no realizado", millones(resumen["no_realizado"]),
                pct(resumen["rend_pct"]), clase(resumen["no_realizado"])),
        tarjeta("P&L realizado", millones(resumen["realizado"]),
                clase_delta=clase(resumen["realizado"]),
                nota="ventas del periodo"),
        tarjeta("Resultado del dia", millones(resumen["pnl_dia"]),
                pct(resumen["var_dia_pct"]), clase(resumen["pnl_dia"])),
        tarjeta("Efectivo", millones(resumen["efectivo"]),
                pct(resumen["peso_efectivo_pct"], 1, False),
                clase(resumen["efectivo"])),
    ], columnas=7)

    # --- Pestanas ---------------------------------------------------------
    tabs = st.tabs(["Resumen", "Posiciones", "Segmentacion", "Riesgo",
                    "Atribucion", "Efectivo", "Operaciones", "Diagnostico"])

    with tabs[0]:
        pestana_resumen(val, res, resumen, hist_activos, bench_hist,
                        riesgo, conc, fx)
    with tabs[1]:
        pestana_posiciones(val)
    with tabs[2]:
        pestana_segmentacion(val, res.efectivo)
    with tabs[3]:
        pestana_riesgo(val, hist_activos, bench_hist, riesgo, conc,
                       riesgo_pos, tecnicos, sin_historico)
    with tabs[4]:
        pestana_atribucion(val, tecnicos)
    with tabs[5]:
        pestana_efectivo(res, resumen, fx)
    with tabs[6]:
        pestana_operaciones(res, cfg)
    with tabs[7]:
        pestana_diagnostico(hallazgos, res)

    st.markdown(
        f'<div class="pie"><b>Punto Casa de Bolsa</b> · Portafolio de Inversion BAZ · '
        f'Fuente de precios: Yahoo Finance (pueden presentar retraso de hasta 15 minutos). '
        f'Documento de trabajo para uso interno; no constituye una recomendacion de '
        f'inversion. Las cifras de costo provienen de «{cfg["fuente_nombre"]}» y de las '
        f'boletas del custodio cargadas en la sesion.</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
