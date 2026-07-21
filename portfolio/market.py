"""
Capa de datos de mercado sobre Yahoo Finance.

Todos los tickers .MX cotizan en pesos, incluidas las emisoras del SIC,
por lo que la valuacion es directa y no requiere conversion cambiaria.
El tipo de cambio se descarga solo para medir exposicion y para el
analisis de sensibilidad FX.

Las descargas se hacen en lote y se memorizan con `st.cache_data` cuando
Streamlit esta disponible, para no golpear la API en cada interaccion.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

try:  # el modulo tambien debe poder usarse fuera de Streamlit
    import streamlit as st

    _cache = st.cache_data
except Exception:  # pragma: no cover
    def _cache(**_kw):
        def deco(fn):
            return fn
        return deco


TTL_INTRADIA = 300      # 5 min para precios vivos
TTL_HISTORICO = 3600    # 1 h para series historicas


def _normalizar_descarga(crudo: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Deja la descarga de yfinance como un DataFrame fecha x ticker de cierres."""
    if crudo is None or not len(crudo):
        return pd.DataFrame()

    if isinstance(crudo.columns, pd.MultiIndex):
        campo = "Close" if "Close" in crudo.columns.get_level_values(0) else None
        cierres = crudo[campo] if campo else crudo.xs("Close", axis=1, level=-1)
    else:
        cierres = crudo[["Close"]] if "Close" in crudo.columns else crudo
        if len(tickers) == 1:
            cierres.columns = tickers

    cierres = cierres.copy()
    cierres.index = pd.to_datetime(cierres.index).tz_localize(None).normalize()
    return cierres.sort_index()


@_cache(ttl=TTL_HISTORICO, show_spinner=False)
def descargar_historico(tickers: tuple[str, ...], dias: int = 400) -> pd.DataFrame:
    """Cierres ajustados de los ultimos `dias` naturales, fecha x ticker."""
    lista = [t for t in dict.fromkeys(tickers) if t]
    if not lista:
        return pd.DataFrame()

    inicio = date.today() - timedelta(days=dias)
    try:
        crudo = yf.download(
            lista, start=inicio, auto_adjust=True, progress=False,
            group_by="column", threads=True,
        )
        df = _normalizar_descarga(crudo, lista)
    except Exception:
        df = pd.DataFrame()

    # Igual que con los precios vigentes: lo que el lote no trajo se reintenta
    # individualmente. Sin esto, un fallo de red se confunde con una emisora
    # sin historico y la deja fuera del bloque de riesgo por el motivo
    # equivocado.
    for t in lista:
        if t in df.columns and df[t].notna().sum() > 0:
            continue
        try:
            s = yf.Ticker(t).history(start=inicio, auto_adjust=True)["Close"].dropna()
        except Exception:
            s = pd.Series(dtype=float)
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            df = df.reindex(df.index.union(s.index)) if len(df) else pd.DataFrame(index=s.index)
            df[t] = s.reindex(df.index)
        elif t not in df.columns:
            df[t] = np.nan

    return df[lista].sort_index()


def _serie_individual(ticker: str) -> pd.Series:
    """Cierres de un solo ticker, por si el lote lo dejo fuera."""
    try:
        s = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
    except Exception:
        return pd.Series(dtype=float)


@_cache(ttl=TTL_INTRADIA, show_spinner=False)
def precios_vigentes(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Ultimo precio y cierre previo por ticker.

    Devuelve columnas: ticker, precio, cierre_previo, var_pct, fecha_precio,
    reintentado.

    Se usa la serie diaria de 5 dias: el ultimo dato es el precio vigente
    (intradia si el mercado esta abierto) y el anterior es el cierre previo,
    que es la base correcta para el resultado del dia.

    La descarga en lote de Yahoo suelta tickers de forma intermitente: la
    emisora existe y cotiza, pero esa peticion en particular no la trajo.
    Cuando pasa, se reintenta individualmente solo con las que faltaron, en
    vez de darlas por no cotizadas y caer al precio de referencia del archivo,
    que es un error silencioso y dificil de notar.
    """
    lista = [t for t in dict.fromkeys(tickers) if t]
    if not lista:
        return pd.DataFrame(columns=["ticker", "precio", "cierre_previo",
                                     "var_pct", "fecha_precio", "reintentado"])

    try:
        crudo = yf.download(lista, period="5d", interval="1d", auto_adjust=False,
                            progress=False, group_by="column", threads=True)
        cierres = _normalizar_descarga(crudo, lista)
    except Exception:
        cierres = pd.DataFrame()

    filas = []
    for t in lista:
        serie = (cierres[t].dropna()
                 if len(cierres) and t in cierres.columns
                 else pd.Series(dtype=float))
        reintentado = False
        if not len(serie):
            serie = _serie_individual(t)
            reintentado = len(serie) > 0

        if not len(serie):
            filas.append(dict(ticker=t, precio=np.nan, cierre_previo=np.nan,
                              var_pct=np.nan, fecha_precio=pd.NaT,
                              reintentado=False))
            continue

        precio = float(serie.iloc[-1])
        previo = float(serie.iloc[-2]) if len(serie) > 1 else np.nan
        filas.append(dict(
            ticker=t,
            precio=precio,
            cierre_previo=previo,
            var_pct=((precio / previo - 1.0) * 100.0
                     if previo and not np.isnan(previo) else np.nan),
            fecha_precio=serie.index[-1],
            reintentado=reintentado,
        ))
    return pd.DataFrame(filas)


def mapas_de_precio(vigentes: pd.DataFrame) -> tuple[dict, dict]:
    """Convierte el DataFrame de precios en los dos diccionarios que usa el motor."""
    if not len(vigentes):
        return {}, {}
    v = vigentes.dropna(subset=["precio"])
    precios = dict(zip(v["ticker"], v["precio"]))
    previos = dict(zip(
        vigentes.dropna(subset=["cierre_previo"])["ticker"],
        vigentes.dropna(subset=["cierre_previo"])["cierre_previo"],
    ))
    return precios, previos


@_cache(ttl=TTL_INTRADIA, show_spinner=False)
def tipo_de_cambio(ticker: str = "USDMXN=X") -> dict:
    """Ultimo USDMXN y su variacion diaria."""
    try:
        serie = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
        if not len(serie):
            return dict(valor=np.nan, var_pct=np.nan)
        valor = float(serie.iloc[-1])
        previo = float(serie.iloc[-2]) if len(serie) > 1 else np.nan
        return dict(valor=valor,
                    var_pct=(valor / previo - 1.0) * 100.0 if previo else np.nan)
    except Exception:
        return dict(valor=np.nan, var_pct=np.nan)


def rendimientos(historico: pd.DataFrame) -> pd.DataFrame:
    """Rendimientos logaritmicos diarios."""
    if not len(historico):
        return pd.DataFrame()
    return np.log(historico / historico.shift(1)).dropna(how="all")


def indicadores_tecnicos(historico: pd.DataFrame) -> pd.DataFrame:
    """
    Senales tecnicas por instrumento: medias moviles, RSI de 14 sesiones,
    distancia al maximo de 52 semanas y momentum de 1 y 3 meses.
    """
    filas = []
    for t in historico.columns:
        s = historico[t].dropna()
        if len(s) < 30:
            continue
        ultimo = float(s.iloc[-1])
        m50 = float(s.tail(50).mean())
        m200 = float(s.tail(200).mean()) if len(s) >= 200 else np.nan
        max52 = float(s.tail(252).max())
        min52 = float(s.tail(252).min())

        delta = s.diff()
        ganancia = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        perdida = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = ganancia / perdida.replace(0, np.nan)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        def mom(n: int) -> float:
            return (ultimo / float(s.iloc[-n]) - 1.0) * 100.0 if len(s) > n else np.nan

        filas.append(dict(
            ticker=t, precio=ultimo,
            vs_media50_pct=(ultimo / m50 - 1.0) * 100.0 if m50 else np.nan,
            vs_media200_pct=(ultimo / m200 - 1.0) * 100.0 if m200 == m200 and m200 else np.nan,
            desde_max52_pct=(ultimo / max52 - 1.0) * 100.0 if max52 else np.nan,
            sobre_min52_pct=(ultimo / min52 - 1.0) * 100.0 if min52 else np.nan,
            rsi14=rsi, momentum_1m_pct=mom(21), momentum_3m_pct=mom(63),
        ))
    return pd.DataFrame(filas)
