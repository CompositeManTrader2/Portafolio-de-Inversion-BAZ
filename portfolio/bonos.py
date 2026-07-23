"""
Analitica de deuda gubernamental sobre el vector analitico de Valmer.

El vector trae precio limpio, interes acumulado, cupon, dias transcurridos
del cupon, cupones por cobrar, duracion y convexidad, pero no las tasas de
rendimiento; aqui se calculan con las convenciones del mercado mexicano:

  CETES     rendimiento de descuento anualizado 360:  y = (VN/P - 1)·360/d
  BONOS M   cupon semestral de 182 dias, formula estandar de mercado:
            sucio = Σ C·182/360 / (1+y·182/360)^(i-f) + 100/(1+y·182/360)^(n-f)
            con f = dias transcurridos/182; se resuelve y por biseccion.
  UDIBONOS  misma formula sobre el precio expresado en UDIs -> tasa real.
            El valor de la UDI se recupera del interes acumulado en pesos:
            UDI = int_acum / (cupon/100 · dias/360 · 100).

El vector se relee solo cuando cambia el archivo (cache por mtime): pesa
~29 MB y tarda ~20 s en parsear, inaceptable por interaccion.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).parent.parent
COLUMNAS = ["FECHA", "TIPO VALOR", "EMISORA", "SERIE", "PRECIO LIMPIO",
            "PRECIO SUCIO", "INTERESES ACUMULADOS", "SOBRETASA", "FECHA VCTO",
            "TASA CUPON", "DIAS TRANSC. CPN", "CUPONES X COBRAR",
            "DURACION", "CONVEXIDAD", "MONTO EN CIRCULACION"]


def ruta_vector() -> Path | None:
    candidatos = sorted(RAIZ.glob("data/Vector*.xls*"))
    return candidatos[-1] if candidatos else None


@lru_cache(maxsize=2)
def _leer(ruta: str, mtime: float) -> pd.DataFrame:
    df = pd.read_excel(ruta, engine="xlrd", usecols=COLUMNAS)
    df["TIPO VALOR"] = df["TIPO VALOR"].astype(str).str.strip()
    return df


def _ytm_bono(sucio: float, cupon: float, n: int, f: float,
              lo: float = -0.05, hi: float = 0.60) -> float:
    """Resuelve la tasa de la formula estandar 182/360 por biseccion."""
    c = cupon * 182.0 / 360.0

    def precio(y):
        u = 1.0 + y * 182.0 / 360.0
        pot = np.arange(1, n + 1) - f
        return float((c / u ** pot).sum() + 100.0 / u ** (n - f))

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if precio(mid) > sucio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cargar(ruta: Path | None = None) -> dict | None:
    ruta = ruta or ruta_vector()
    if ruta is None or not ruta.exists():
        return None
    df = _leer(str(ruta), ruta.stat().st_mtime)

    fecha = datetime.strptime(str(int(df["FECHA"].iloc[0])), "%Y%m%d").date()

    def _dias(vcto) -> int:
        return (pd.Timestamp(vcto).date() - fecha).days

    # ---- CETES -----------------------------------------------------------
    bi = df[df["TIPO VALOR"] == "BI"].copy()
    bi["dias"] = bi["FECHA VCTO"].map(_dias)
    bi = bi[bi["dias"] > 3].sort_values("dias")
    bi["anios"] = bi["dias"] / 365.0
    bi["ytm"] = (10.0 / bi["PRECIO LIMPIO"] - 1.0) * 360.0 / bi["dias"] * 100.0

    # ---- BONOS M ---------------------------------------------------------
    m = df[df["TIPO VALOR"] == "M"].copy()
    m["dias"] = m["FECHA VCTO"].map(_dias)
    m = m[m["dias"] > 30].sort_values("dias")
    m["anios"] = m["dias"] / 365.25
    m["sucio"] = m["PRECIO LIMPIO"] + m["INTERESES ACUMULADOS"]
    m["ytm"] = [
        _ytm_bono(su, cu, int(n_), dt / 182.0) * 100.0
        for su, cu, n_, dt in zip(m["sucio"], m["TASA CUPON"],
                                  m["CUPONES X COBRAR"],
                                  m["DIAS TRANSC. CPN"])]

    # ---- UDIBONOS (tasa real) -------------------------------------------
    s = df[df["TIPO VALOR"] == "S"].copy()
    s["dias"] = s["FECHA VCTO"].map(_dias)
    s = s[s["dias"] > 30].sort_values("dias")
    s["anios"] = s["dias"] / 365.25
    # UDI implicita por bono; la mediana filtra redondeos
    udi_por_bono = (s["INTERESES ACUMULADOS"]
                    / (s["TASA CUPON"] / 100.0 * s["DIAS TRANSC. CPN"]
                       / 360.0 * 100.0))
    udi = float(udi_por_bono.median())
    s["sucio_udis"] = (s["PRECIO LIMPIO"] + s["INTERESES ACUMULADOS"]) / udi
    s["ytm"] = [
        _ytm_bono(su, cu, int(n_), dt / 182.0) * 100.0
        for su, cu, n_, dt in zip(s["sucio_udis"], s["TASA CUPON"],
                                  s["CUPONES X COBRAR"],
                                  s["DIAS TRANSC. CPN"])]

    # ---- BONDES F (sobretasa) -------------------------------------------
    lf = df[df["TIPO VALOR"] == "LF"].copy()
    lf["dias"] = lf["FECHA VCTO"].map(_dias)
    lf = lf[lf["dias"] > 3].sort_values("dias")
    lf["anios"] = lf["dias"] / 365.0

    return dict(fecha=fecha, udi=udi, cetes=bi, bonos_m=m,
                udibonos=s, bondesf=lf)


# --------------------------------------------------------------------------
# Curvas y derivados
# --------------------------------------------------------------------------

def _interp(anios: np.ndarray, ytms: np.ndarray, t: float) -> float:
    """Interpolacion lineal acotada sobre la curva (anios, ytm)."""
    return float(np.interp(t, anios, ytms))


def curva_nominal(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """CETES (tramo corto) + Bonos M en una sola curva nominal."""
    bi, m = d["cetes"], d["bonos_m"]
    anios = np.concatenate([bi["anios"].values, m["anios"].values])
    ytms = np.concatenate([bi["ytm"].values, m["ytm"].values])
    orden = np.argsort(anios)
    return anios[orden], ytms[orden]


def nodo_cercano(df: pd.DataFrame, anios_objetivo: float) -> pd.Series:
    return df.iloc[(df["anios"] - anios_objetivo).abs().argmin()]


def carry_rolldown(d: dict, horizonte_anios: float = 0.25) -> pd.DataFrame:
    """
    Carry y rolldown a 3 meses por nodo de la curva M, en puntos base de
    precio, fondeado al CETE mas cercano a 91 dias.

      carry_bp    = (ytm - fondeo) · h · 100
      rolldown_bp = (ytm - ytm_interp(t - h)) · duracion · 100

    El rolldown supone curva sin cambios: el bono "rueda" hacia un punto
    mas corto de la curva y gana (o pierde) esa diferencia de tasa por su
    duracion.
    """
    fondeo = float(nodo_cercano(d["cetes"], 91 / 365.0)["ytm"])
    an_, yt_ = curva_nominal(d)
    m = d["bonos_m"]
    filas = []
    for serie, anios, ytm, cupon, dur in zip(
            m["SERIE"], m["anios"], m["ytm"], m["TASA CUPON"], m["DURACION"]):
        y_fut = _interp(an_, yt_, max(anios - horizonte_anios, 0.02))
        carry = (ytm - fondeo) * horizonte_anios * 100.0
        roll = (ytm - y_fut) * dur * 100.0
        filas.append(dict(serie=str(serie), anios=anios, ytm=ytm,
                          cupon=cupon, carry_bp=carry, roll_bp=roll,
                          total_bp=carry + roll))
    out = pd.DataFrame(filas)
    out["fondeo"] = fondeo
    return out


def analitica(d: dict) -> dict:
    """Niveles, pendientes, mariposa y breakevens de la sesion."""
    bi, m, s = d["cetes"], d["bonos_m"], d["udibonos"]
    an_m, yt_m = m["anios"].values, m["ytm"].values
    an_s, yt_s = s["anios"].values, s["ytm"].values

    cete28 = float(nodo_cercano(bi, 28 / 365.0)["ytm"])
    cete91 = float(nodo_cercano(bi, 91 / 365.0)["ytm"])
    cete364 = float(nodo_cercano(bi, 1.0)["ytm"])
    m2 = _interp(an_m, yt_m, 2.0)
    m5 = _interp(an_m, yt_m, 5.0)
    m10 = _interp(an_m, yt_m, 10.0)
    m30 = _interp(an_m, yt_m, min(30.0, float(an_m.max())))
    s10 = _interp(an_s, yt_s, 10.0)
    s3 = _interp(an_s, yt_s, 3.0)
    m3 = _interp(an_m, yt_m, 3.0)

    nodo_m2 = nodo_cercano(m, 2.0)

    return dict(
        cete28=cete28, cete91=cete91, cete364=cete364,
        m2=m2, m5=m5, m10=m10, m30=m30,
        p2s10=(m10 - m2) * 100.0, p10s30=(m30 - m10) * 100.0,
        ext_corta=(float(nodo_m2["ytm"]) - cete364) * 100.0,
        nodo_m2=nodo_m2,
        fly_2_5_10=(2.0 * m5 - m2 - m10) * 100.0,
        be3=(m3 - s3) * 100.0 / 100.0, be10=(m10 - s10) * 100.0 / 100.0,
        sobretasa_lf=float(d["bondesf"]["SOBRETASA"].mean() * 100.0),
    )
