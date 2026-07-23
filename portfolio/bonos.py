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
CSV_CRED = RAIZ / "data" / "vector_credito.csv"
CSV_CORP = RAIZ / "data" / "vector_corp.csv"

# tipos de valor corporativos que la cartera puede tener en posicion
TIPOS_CORP_POS = ("90", "91", "93", "94", "95")
DIR_HIST = RAIZ / "data" / "hist_gob"

# corporativos de tasa revisable para las curvas de credito
TIPOS_CREDITO = ("91", "93", "94", "95", "D1", "D2", "OD")
COLS_CREDITO = ["FECHA", "TIPO VALOR", "EMISORA", "SERIE", "SOBRETASA",
                "FECHA VCTO", "S&P", "MDYS", "CALIFICACION FITCH",
                "HR RATINGS", "SECTOR", "MONTO EN CIRCULACION",
                "BURSATILIDAD"]


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
    # persistir el universo corporativo valuable (para posiciones propias)
    corp = df[df["TIPO VALOR"].isin(TIPOS_CORP_POS)]
    try:
        corp.to_csv(CSV_CORP, index=False)
    except OSError:
        pass
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
            # foto diaria para el historico de curvas (una por fecha de vector)
            fecha = str(int(gob["FECHA"].iloc[0]))
            DIR_HIST.mkdir(exist_ok=True)
            destino_hist = DIR_HIST / f"gob_{fecha}.csv"
            if not destino_hist.exists():
                gob.to_csv(destino_hist, index=False)
            _destilar_credito(xls)
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


def cargar_corp() -> pd.DataFrame | None:
    """Universo corporativo destilado (tipos 90/91/93/94/95) para valuar
    posiciones propias; None si aun no se destila de un vector."""
    if not CSV_CORP.exists():
        return None
    df = _leer_csv(str(CSV_CORP), CSV_CORP.stat().st_mtime)
    df["oficial"] = pd.to_numeric(df["TASA DE RENDIMIENTO"], errors="coerce")
    df["ytm"] = df["oficial"]
    return df


def cargar(ruta: Path | None = None) -> dict | None:
    df = _obtener_gubernamentales(ruta)
    if df is None or not len(df):
        return None
    return _procesar(df)


def _procesar(df: pd.DataFrame) -> dict:
    fecha = datetime.strptime(str(int(df["FECHA"].iloc[0])), "%Y%m%d").date()

    def _dias(vcto) -> int:
        return (pd.Timestamp(vcto).date() - fecha).days

    # ---- CETES -----------------------------------------------------------
    bi = df[df["TIPO VALOR"] == "BI"].copy()
    bi["dias"] = bi["FECHA VCTO"].map(_dias)
    bi = bi[bi["dias"] > 0].sort_values("dias")
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


# --------------------------------------------------------------------------
# Tasa revisable: sobretasas de BONDES F y fijo contra flotante
# --------------------------------------------------------------------------

def sobretasas_bondesf(d: dict) -> pd.DataFrame:
    """Curva de sobretasas de BONDES F (pb) por plazo."""
    lf = d["bondesf"].copy()
    lf["st_pb"] = pd.to_numeric(lf["SOBRETASA"], errors="coerce") * 100.0
    lf = lf.dropna(subset=["st_pb"]).sort_values("anios")
    return lf[["SERIE", "anios", "st_pb", "ytm"]]


def _st_interp(lf: pd.DataFrame, t: float) -> float:
    return float(np.interp(t, lf["anios"].values, lf["st_pb"].values))


def fijo_vs_flotante(d: dict) -> pd.DataFrame:
    """
    Breakeven de fondeo por plazo: el BONDES F paga fondeo + sobretasa y el
    Bono M paga fijo, asi que el flotante gana si el fondeo PROMEDIO del
    periodo supera (fija - sobretasa). La ultima columna traduce eso a pb de
    alza promedio requerida desde el fondeo de hoy.
    """
    lf = sobretasas_bondesf(d)
    an_, yt_ = curva_nominal(d)
    fondeo = float(nodo_cercano(d["cetes"], 91 / 365.0)["ytm"])
    filas = []
    for t in (1.0, 1.61, 2.61, 3.61, 4.85):
        y_fijo = _interp(an_, yt_, t)
        st = _st_interp(lf, min(t, float(lf["anios"].max())))
        breakeven = y_fijo - st / 100.0
        filas.append(dict(
            anios=t, fijo=y_fijo, sobretasa_pb=st,
            breakeven_fondeo=breakeven,
            alza_requerida_pb=(breakeven - fondeo) * 100.0))
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Credito corporativo (tasa revisable): sobretasas por calificacion
# --------------------------------------------------------------------------

def _destilar_credito(xls: Path) -> None:
    """Subset corporativo del vector -> data/vector_credito.csv (~200 KB)."""
    motor = "xlrd" if str(xls).lower().endswith(".xls") else "openpyxl"
    df = pd.read_excel(xls, engine=motor, usecols=COLS_CREDITO)
    df["TIPO VALOR"] = df["TIPO VALOR"].astype(str).str.strip()
    df["st"] = pd.to_numeric(df["SOBRETASA"], errors="coerce")
    df = df[df["TIPO VALOR"].isin(TIPOS_CREDITO) & (df["st"] > 0.001)]
    df.drop(columns=["st"]).to_csv(CSV_CRED, index=False)


def _bucket_rating(fila) -> str:
    for c in ("S&P", "MDYS", "CALIFICACION FITCH", "HR RATINGS"):
        v = str(fila.get(c, "")).strip().upper()
        if v and v not in ("-", "NAN", "N.D.", "ND", ""):
            v = v.replace("MX", "").replace("HR", "").replace(" ", "")
            if v.startswith("AAA"):
                return "AAA"
            if v.startswith("AA"):
                return "AA"
            if v.startswith("A"):
                return "A"
            if v.startswith("BBB"):
                return "BBB"
            return "Sub-BBB/otro"
    return "Sin calificacion"


def credito_buckets() -> dict | None:
    """
    Sobretasa mediana (pb) por escalon de calificacion x plazo con conteos.
    Mediana y no promedio: el universo mezcla quirografarios, bancarios y
    estructurados y las colas contaminan el promedio (curaduria por sector
    pendiente; la nota lo declara).
    """
    if not CSV_CRED.exists():
        return None
    df = pd.read_csv(CSV_CRED)
    fecha = datetime.strptime(str(int(df["FECHA"].iloc[0])), "%Y%m%d").date()
    df["st_pb"] = pd.to_numeric(df["SOBRETASA"], errors="coerce") * 100.0
    df["anios"] = (pd.to_datetime(df["FECHA VCTO"])
                   - pd.Timestamp(fecha)).dt.days / 365.0
    df = df.dropna(subset=["st_pb"])
    df = df[(df["anios"] > 0.05) & (df["st_pb"] < 1000)]
    df["rating"] = df.apply(_bucket_rating, axis=1)
    df["plazo"] = pd.cut(df["anios"], [0, 1, 3, 5, 30],
                         labels=["0-1a", "1-3a", "3-5a", "5a+"])

    orden = ["AAA", "AA", "A", "BBB"]
    tabla = []
    for r in orden:
        sub = df[df["rating"] == r]
        fila = dict(rating=r, n=int(len(sub)))
        for pz in ["0-1a", "1-3a", "3-5a", "5a+"]:
            g = sub[sub["plazo"] == pz]["st_pb"]
            fila[pz] = (float(g.median()), int(len(g))) if len(g) >= 3 else None
        tabla.append(fila)

    med = {}
    for r in orden:
        serie = df[df["rating"] == r]["st_pb"]
        if len(serie) >= 3:
            med[r] = float(serie.median())
    pickups = []
    for a, b in zip(orden, orden[1:]):
        if a in med and b in med:
            pickups.append((a + " a " + b, med[b] - med[a]))
    return dict(fecha=fecha, tabla=tabla, pickups=pickups,
                total=int(len(df)),
                sin_calif=int((df["rating"] == "Sin calificacion").sum()))


# --------------------------------------------------------------------------
# Historico de vectores: metricas por sesion y percentiles
# --------------------------------------------------------------------------

def historico_metricas() -> pd.DataFrame:
    """Una fila de metricas de curva por cada snapshot en data/hist_gob."""
    if not DIR_HIST.exists():
        return pd.DataFrame()
    filas = []
    for f in sorted(DIR_HIST.glob("gob_*.csv")):
        try:
            d = _procesar(pd.read_csv(f))
            a = analitica(d)
            filas.append(dict(
                fecha=d["fecha"], cete364=a["cete364"], m2=a["m2"],
                m5=a["m5"], m10=a["m10"], p2s10=a["p2s10"],
                p10s30=a["p10s30"], fly=a["fly_2_5_10"],
                ext_corta=a["ext_corta"], be10=a["be10"]))
        except Exception:
            continue
    return pd.DataFrame(filas)


def percentiles_hoy(hist: pd.DataFrame, a: dict) -> list[dict]:
    """Donde cae cada metrica de hoy dentro de su propia historia."""
    mapa = [("2s10s", "p2s10", a["p2s10"], "pb"),
            ("Fly 2s5s10s", "fly", a["fly_2_5_10"], "pb"),
            ("Extension M2-CETE1a", "ext_corta", a["ext_corta"], "pb"),
            ("BE 10a", "be10", a["be10"], "%"),
            ("M10", "m10", a["m10"], "%"),
            ("CETE 1a", "cete364", a["cete364"], "%")]
    out = []
    for etiqueta, col, hoy, unidad in mapa:
        serie = hist[col].dropna() if col in hist else pd.Series(dtype=float)
        pct = (float((serie <= hoy).mean() * 100.0)
               if len(serie) >= 5 else None)
        out.append(dict(
            etiqueta=etiqueta, hoy=float(hoy), unidad=unidad, pct=pct,
            minimo=float(serie.min()) if len(serie) else None,
            maximo=float(serie.max()) if len(serie) else None,
            n=int(len(serie))))
    return out


# --------------------------------------------------------------------------
# Udibonos en escala nominal: la curva real transformada por escenario de
# inflacion, comparable uno a uno contra la curva de Bonos M
# --------------------------------------------------------------------------

def udibonos_nominalizados(d: dict,
                           escenarios=(3.0, 3.5, 4.0)) -> pd.DataFrame:
    """
    Cada udibono llevado a tasa nominal equivalente bajo escenarios de
    inflacion promedio, con Fisher exacto:

        y_nominal_eq = (1 + y_real)·(1 + pi) − 1

    y comparado contra el Bono M interpolado al mismo plazo. La columna
    be_nodo es la inflacion de indiferencia del nodo ((1+y_M)/(1+y_real)−1):
    si tu escenario de inflacion promedio supera ese numero, el udibono gana
    al nominal en ese plazo; si no, pierde. La ventaja por escenario es la
    diferencia en pb contra el M del mismo plazo.
    """
    an_, yt_ = curva_nominal(d)
    s = d["udibonos"]
    filas = []
    for serie, anios, y_real in zip(s["SERIE"], s["anios"], s["ytm"]):
        y_m = _interp(an_, yt_, float(anios))
        fila = dict(
            serie=str(serie), anios=float(anios), real=float(y_real),
            m_interp=y_m,
            be_nodo=((1 + y_m / 100) / (1 + y_real / 100) - 1) * 100)
        for pi in escenarios:
            nom = ((1 + y_real / 100) * (1 + pi / 100) - 1) * 100
            fila[f"nom_{pi:.1f}"] = nom
            fila[f"vent_{pi:.1f}"] = (nom - y_m) * 100.0   # pb vs M
        filas.append(fila)
    return pd.DataFrame(filas)
