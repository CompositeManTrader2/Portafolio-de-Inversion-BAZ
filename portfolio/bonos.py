"""
Analitica de deuda gubernamental sobre el vector analitico de Valmer.

La columna TASA DE RENDIMIENTO del vector es la fuente primaria de tasas.
Ademas, cada carga recalcula las tasas desde el precio con las convenciones
del mercado mexicano y las compara contra las oficiales: si la discrepancia
maxima excede la tolerancia, el modulo la reporta en `validacion` (y si la
columna oficial faltara en un vector futuro, el solver toma su lugar).
Convenciones del solver:

  CETES     rendimiento de descuento anualizado 360:  y = (VN/P - 1)·360/d
  BONOS M   cupon semestral de 182 dias, formula estandar de mercado:
            sucio = Σ C·182/360 / (1+y·182/360)^(i-f) + 100/(1+y·182/360)^(n-f)
            con f = dias transcurridos/182; se resuelve y por biseccion.
  UDIBONOS  misma formula sobre el precio expresado en UDIs -> tasa real.
            El valor de la UDI se recupera del interes acumulado en pesos:
            UDI = int_acum / (cupon/100 · dias/360 · 100).

Duraciones del vector: DURACION MACAULAY viene en DIAS; DURACION es la
duracion modificada en anios base 360 (verificado: macaulay/(1+y·182/360)/360
reproduce la columna), que es la correcta para DV01 y rolldown.

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
            "DURACION", "DURACION MACAULAY", "CONVEXIDAD",
            "MONTO EN CIRCULACION", "TASA DE RENDIMIENTO"]


GUBERNAMENTALES = ("BI", "M", "S", "LF", "LD")
CSV_GOB = RAIZ / "data" / "vector_gob.csv"


def ruta_vector() -> Path | None:
    """El vector .xls vigente es el mas reciente por fecha de archivo, de
    modo que uno subido desde la app releva al que viaja en disco."""
    candidatos = list(RAIZ.glob("data/Vector*.xls*"))
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


@lru_cache(maxsize=2)
def _leer_xls(ruta: str, mtime: float) -> pd.DataFrame:
    motor = "xlrd" if ruta.lower().endswith(".xls") else "openpyxl"
    df = pd.read_excel(ruta, engine=motor, usecols=COLUMNAS)
    df["TIPO VALOR"] = df["TIPO VALOR"].astype(str).str.strip()
    return df[df["TIPO VALOR"].isin(GUBERNAMENTALES)].reset_index(drop=True)


@lru_cache(maxsize=2)
def _leer_csv(ruta: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df["TIPO VALOR"] = df["TIPO VALOR"].astype(str).str.strip()
    return df


def _obtener_gubernamentales(ruta_xls: Path | None = None) -> pd.DataFrame | None:
    """
    Los ~140 renglones gubernamentales del vector, por la via mas barata.

    El .xls completo pesa ~29 MB y parsearlo tarda decenas de segundos y
    cientos de MB de memoria: inviable en cada arranque (y letal en Cloud).
    Por eso, la primera vez que aparece un .xls mas nuevo que el destilado
    se parsea UNA vez y se persiste data/vector_gob.csv (~30 KB); todas las
    cargas posteriores leen el CSV. El CSV viaja en el repositorio, asi que
    el despliegue nunca necesita el archivo gigante ni el motor xlrd.
    """
    xls = ruta_xls or ruta_vector()
    csv_mtime = CSV_GOB.stat().st_mtime if CSV_GOB.exists() else -1.0

    if xls is not None and xls.exists() and xls.stat().st_mtime > csv_mtime:
        gob = _leer_xls(str(xls), xls.stat().st_mtime)
        try:
            gob.to_csv(CSV_GOB, index=False)
        except OSError:
            pass
        return gob
    if CSV_GOB.exists():
        return _leer_csv(str(CSV_GOB), CSV_GOB.stat().st_mtime)
    return None


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
    df = _obtener_gubernamentales(ruta)
    if df is None or not len(df):
        return None

    fecha = datetime.strptime(str(int(df["FECHA"].iloc[0])), "%Y%m%d").date()

    def _dias(vcto) -> int:
        return (pd.Timestamp(vcto).date() - fecha).days

    # ---- CETES -----------------------------------------------------------
    bi = df[df["TIPO VALOR"] == "BI"].copy()
    bi["dias"] = bi["FECHA VCTO"].map(_dias)
    bi = bi[bi["dias"] > 3].sort_values("dias")
    bi["anios"] = bi["dias"] / 365.0
    bi["ytm_solver"] = (10.0 / bi["PRECIO LIMPIO"] - 1.0) * 360.0 / bi["dias"] * 100.0
    bi["oficial"] = pd.to_numeric(bi["TASA DE RENDIMIENTO"], errors="coerce")
    bi["ytm"] = bi["oficial"].fillna(bi["ytm_solver"])

    # ---- BONOS M ---------------------------------------------------------
    m = df[df["TIPO VALOR"] == "M"].copy()
    m["dias"] = m["FECHA VCTO"].map(_dias)
    m = m[m["dias"] > 30].sort_values("dias")
    m["anios"] = m["dias"] / 365.25
    m["sucio"] = m["PRECIO LIMPIO"] + m["INTERESES ACUMULADOS"]
    m["ytm_solver"] = [
        _ytm_bono(su, cu, int(n_), dt / 182.0) * 100.0
        for su, cu, n_, dt in zip(m["sucio"], m["TASA CUPON"],
                                  m["CUPONES X COBRAR"],
                                  m["DIAS TRANSC. CPN"])]
    m["oficial"] = pd.to_numeric(m["TASA DE RENDIMIENTO"], errors="coerce")
    m["ytm"] = m["oficial"].fillna(m["ytm_solver"])

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
    s["ytm_solver"] = [
        _ytm_bono(su, cu, int(n_), dt / 182.0) * 100.0
        for su, cu, n_, dt in zip(s["sucio_udis"], s["TASA CUPON"],
                                  s["CUPONES X COBRAR"],
                                  s["DIAS TRANSC. CPN"])]
    s["oficial"] = pd.to_numeric(s["TASA DE RENDIMIENTO"], errors="coerce")
    s["ytm"] = s["oficial"].fillna(s["ytm_solver"])

    # ---- BONDES F (sobretasa) -------------------------------------------
    lf = df[df["TIPO VALOR"] == "LF"].copy()
    lf["dias"] = lf["FECHA VCTO"].map(_dias)
    lf = lf[lf["dias"] > 3].sort_values("dias")
    lf["anios"] = lf["dias"] / 365.0

    def _difmax(df_):
        con = df_.dropna(subset=["oficial"])
        if not len(con):
            return None
        return float((con["ytm_solver"] - con["oficial"]).abs().max() * 100.0)

    validacion = dict(cetes_pb=_difmax(bi), m_pb=_difmax(m),
                      udibonos_pb=_difmax(s))
    validacion["max_pb"] = max(v for v in validacion.values()
                               if v is not None)

    lf["ytm"] = pd.to_numeric(lf["TASA DE RENDIMIENTO"], errors="coerce")

    return dict(fecha=fecha, udi=udi, cetes=bi, bonos_m=m,
                udibonos=s, bondesf=lf, validacion=validacion)


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


def residuos_curva(d: dict) -> pd.DataFrame:
    """
    Rich/cheap por Bono M: residuo en pb contra una curva suave (polinomio
    cubico en raiz del plazo). Positivo = barato (rinde de mas), negativo =
    caro. Es la base del valor relativo intra-curva.
    """
    m = d["bonos_m"]
    x = m["anios"].values.astype(float)
    y = m["ytm"].values.astype(float)
    coef = np.polyfit(np.sqrt(x), y, 3)
    ajuste = np.polyval(coef, np.sqrt(x))
    return pd.DataFrame(dict(
        serie=m["SERIE"].astype(str).values, anios=x, ytm=y,
        residuo_pb=(y - ajuste) * 100.0))


def forwards_clave(d: dict) -> list[dict]:
    """Tasas forward implicitas entre nodos de la curva nominal."""
    an_, yt_ = curva_nominal(d)

    def z(t):
        return _interp(an_, yt_, t)

    out = []
    for t1, t2 in [(1, 2), (2, 3), (2, 5), (5, 10), (10, 20)]:
        y1, y2 = z(t1) / 100.0, z(t2) / 100.0
        fwd = ((1 + y2) ** t2 / (1 + y1) ** t1) ** (1 / (t2 - t1)) - 1
        out.append(dict(t1=t1, t2=t2, spot1=z(t1), spot2=z(t2),
                        fwd=fwd * 100.0))
    return out


def matriz_escenarios(d: dict, choques_pb=(-100, -50, 0, 50, 100),
                      horizonte: float = 1.0) -> pd.DataFrame:
    """
    Retorno total estimado a 12 meses por Bono M bajo choques paralelos:

      TR% = carry (≈ ytm·h) + rolldown (Δy de rodar por la curva · dur)
            + precio del choque (−dur·Δy + ½·convexidad·Δy²)

    La referencia de decision es el CETE de 1 año: la matriz responde
    "¿en qué escenario me gana cada bono contra quedarme en el CETE?".
    """
    an_, yt_ = curva_nominal(d)
    m = d["bonos_m"]
    filas = []
    for serie, anios, ytm, dur, conv in zip(
            m["SERIE"], m["anios"], m["ytm"], m["DURACION"],
            m["CONVEXIDAD"]):
        y_roll = _interp(an_, yt_, max(anios - horizonte, 0.02))
        base = ytm * horizonte + (ytm - y_roll) * dur
        fila = dict(serie=str(serie), anios=float(anios), ytm=float(ytm))
        for c in choques_pb:
            dy = c / 10_000.0
            precio = (-dur * dy + 0.5 * float(conv) * dy * dy) * 100.0
            fila[f"tr_{c:+d}"] = base + precio
        filas.append(fila)
    return pd.DataFrame(filas)
