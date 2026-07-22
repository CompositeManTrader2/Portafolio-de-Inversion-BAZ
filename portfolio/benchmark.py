"""
Benchmark de referencia para la atribucion: S&P/BMV IPC reconstruido a nivel
constituyente.

Los pesos viven en data/benchmark_ipc.csv y son una APROXIMACION del
free-float del indice: S&P no publica los pesos exactos en tiempo real y el
indice rebalancea cada trimestre. El archivo es deliberadamente editable para
refrescarlo con el factsheet oficial; la interfaz declara esta condicion.

La atribucion usa el rendimiento del benchmark reconstruido (suma ponderada
de sus constituyentes), no el nivel del ^MXX, para que la identidad de
Brinson (asignacion + seleccion + interaccion = retorno activo) cierre de
forma exacta. La diferencia contra el ^MXX observado se muestra aparte como
error de reconstruccion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUTA_DEFAULT = Path(__file__).parent.parent / "data" / "benchmark_ipc.csv"


def cargar_benchmark(ruta: Path | str = RUTA_DEFAULT) -> pd.DataFrame:
    """Constituyentes del benchmark con pesos normalizados a 100."""
    df = pd.read_csv(ruta)
    df["peso_pct"] = df["peso_pct"] / df["peso_pct"].sum() * 100.0
    return df


def tickers_benchmark(bench: pd.DataFrame) -> tuple[str, ...]:
    return tuple(bench["ticker"].tolist())


def rendimiento_periodo(historico: pd.DataFrame, ticker: str,
                        sesiones: int) -> float | None:
    """Rendimiento total simple de un ticker en las ultimas `sesiones`."""
    if ticker not in historico.columns:
        return None
    s = historico[ticker].dropna()
    if len(s) < 2:
        return None
    ventana = s.tail(sesiones + 1)
    if len(ventana) < 2 or ventana.iloc[0] == 0:
        return None
    return float(ventana.iloc[-1] / ventana.iloc[0] - 1.0)


def sectores_benchmark(bench: pd.DataFrame, historico: pd.DataFrame,
                       sesiones: int) -> pd.DataFrame:
    """
    Peso y rendimiento del periodo por sector del benchmark.

    El rendimiento sectorial es la suma ponderada de los constituyentes con
    precio disponible; si a un sector le falta algun nombre, el peso del
    sector se re-normaliza sobre los que si tienen dato y la columna
    `cobertura_pct` deja constancia de cuanto peso quedo representado.
    """
    filas = []
    for sector, grupo in bench.groupby("sector"):
        w_total = float(grupo["peso_pct"].sum())
        con_dato = []
        for _, c in grupo.iterrows():
            r = rendimiento_periodo(historico, c["ticker"], sesiones)
            if r is not None:
                con_dato.append((float(c["peso_pct"]), r))
        if not con_dato:
            continue
        w_cubierto = sum(w for w, _ in con_dato)
        r_sector = sum(w * r for w, r in con_dato) / w_cubierto
        filas.append(dict(sector=sector, peso_pct=w_total,
                          rendimiento_pct=r_sector * 100.0,
                          cobertura_pct=w_cubierto / w_total * 100.0))
    return pd.DataFrame(filas)
