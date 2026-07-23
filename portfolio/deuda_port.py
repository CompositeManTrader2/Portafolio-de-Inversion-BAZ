"""
Cartera de renta fija: posiciones, riesgo propio, benchmark y ALM.

Cuatro piezas que convierten la analitica de mercado en gestion de cartera:

1. POSICIONES  data/posiciones_deuda.csv registra boletas de deuda
   (fecha, operacion C/V, tipo_valor, serie, titulos, precio_sucio por
   titulo). Se netean por instrumento con costo promedio ponderado en
   compras (mismo criterio que el motor de capitales) y se valuan contra el
   vector del dia con PRECIO SUCIO oficial. DV01 monetario por posicion =
   duracion modificada x valor x 0.0001.

2. KRD (key-rate DV01)  el DV01 de cada posicion se reparte linealmente
   entre las dos cubetas clave que la flanquean (2/5/10/20/30 anios), el
   estandar de mesa para perfilar la curva.

3. BENCHMARK GUBERNAMENTAL  indice proxy de Bonos M ponderado por MONTO EN
   CIRCULACION del vector, con su perfil KRD normalizado: contra el se mide
   el posicionamiento activo de DV01 por cubeta (la foto que exige un
   mandato).

4. ALM REAL  el pasivo (flujos reales anuales en data/pasivo_real.csv,
   pesos de hoy) se descuenta con la curva real de Udibonos; se comparan
   KRD reales de activo (udibonos en cartera) contra pasivo, y una cartera
   replicante por minimos cuadrados no negativos sugiere el calce.

Nada de esto es recomendacion de inversion; son herramientas de analisis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import bonos as bn

RAIZ = Path(__file__).parent.parent
CSV_POSICIONES = RAIZ / "data" / "posiciones_deuda.csv"
CSV_PASIVO = RAIZ / "data" / "pasivo_real.csv"

CUBETAS = (2.0, 5.0, 10.0, 20.0, 30.0)


# --------------------------------------------------------------------------
# 1. Posiciones
# --------------------------------------------------------------------------

def _pos_xlsx_reciente() -> Path | None:
    candidatos = list((RAIZ / "data").glob("Pos*.xlsx"))
    return max(candidatos, key=lambda x: x.stat().st_mtime) if candidatos else None


def convertir_posicion(ruta_xlsx: Path) -> pd.DataFrame:
    """
    Convierte el estado de posicion de la mesa (formato Pos.xlsx: Operacion,
    Ticker TIPO_EMISORA_SERIE, No. Titulos, Valor Costo Actualizado, Valor
    Mercado, Posicion 24/48/73/96) al registro canonico de boletas. Guarda:
      - costo unitario = Valor Costo Actualizado / titulos (los papeles en
        UDIs ya vienen actualizados por el custodio)
      - vm_ref = Valor Mercado del custodio, para validar nuestra valuacion
      - por_liquidar = neto de las posiciones a 24/48/72/96 horas
      - clase_op = Bono | Repo (el reporto se trata como liquidez colateral,
        sin DV01)
    """
    df = pd.read_excel(ruta_xlsx)
    filas = []
    for r in df.itertuples():
        partes = str(r.Ticker).strip().split("_")
        if len(partes) != 3:
            continue
        tv, emisora, serie = (x.strip() for x in partes)
        titulos = float(r._4)          # No. Titulos
        costo = float(r._6)            # Valor Costo Actualizado
        vm = float(r._7)               # Valor Mercado
        por_liq = sum(float(x) for x in (r._12, r._13, r._14, r._15))
        clase_op = str(r.Operacion).strip()
        precio = (costo / titulos) if (titulos and clase_op != "Repo") else (
            vm / titulos if titulos else 0.0)
        filas.append(dict(
            fecha="", operacion="C", tipo_valor=tv, emisora=emisora,
            serie=serie, titulos=titulos, precio_sucio=precio,
            por_liquidar=por_liq, vm_ref=vm, clase_op=clase_op))
    out = pd.DataFrame(filas)
    out.to_csv(CSV_POSICIONES, index=False)
    return out


def cargar_posiciones() -> pd.DataFrame:
    """Boletas netas por instrumento con costo promedio ponderado. Si hay
    un data/Pos*.xlsx mas nuevo que el CSV, se convierte automaticamente."""
    px = _pos_xlsx_reciente()
    if px is not None and (not CSV_POSICIONES.exists()
                           or px.stat().st_mtime
                           > CSV_POSICIONES.stat().st_mtime):
        convertir_posicion(px)
    if not CSV_POSICIONES.exists():
        return pd.DataFrame()
    b = pd.read_csv(CSV_POSICIONES, dtype={"serie": str})
    b["operacion"] = b["operacion"].astype(str).str.upper().str.strip()
    if "emisora" not in b.columns:      # compatibilidad con el esquema previo
        b["emisora"] = ""
    for c, d in (("por_liquidar", 0.0), ("vm_ref", float("nan")),
                 ("clase_op", "Bono")):
        if c not in b.columns:
            b[c] = d
    netas = []
    for (tv, emi, serie), g in b.groupby(
            ["tipo_valor", "emisora", "serie"], sort=False, dropna=False):
        titulos = costo = 0.0
        for r in g.itertuples():
            q = float(r.titulos)
            if r.operacion.startswith("C"):
                titulos += q
                costo += q * float(r.precio_sucio)
            else:
                if titulos > 0:
                    costo *= max(titulos - q, 0.0) / titulos
                titulos -= q
        if abs(titulos) > 1e-9:
            netas.append(dict(
                tipo_valor=str(tv).strip(), emisora=str(emi).strip(),
                serie=str(serie), titulos=titulos, costo_total=costo,
                costo_unit=costo / titulos,
                por_liquidar=float(g["por_liquidar"].sum()),
                vm_ref=float(g["vm_ref"].sum()),
                clase_op=str(g["clase_op"].iloc[0])))
    return pd.DataFrame(netas)


def _universo(dv: dict) -> pd.DataFrame:
    """Gubernamentales del vector + corporativos destilados, valuables."""
    bloques = []
    for clase, df in (("BI", dv["cetes"]), ("M", dv["bonos_m"]),
                      ("S", dv["udibonos"]), ("LF", dv["bondesf"])):
        d = df.copy()
        d["clase"] = clase
        bloques.append(d)
    corp = bn.cargar_corp()
    if corp is not None and len(corp):
        c = corp.copy()
        c["clase"] = c["TIPO VALOR"]
        fecha = pd.Timestamp(dv["fecha"])
        c["anios"] = ((pd.to_datetime(c["FECHA VCTO"], errors="coerce")
                       - fecha).dt.days / 365.0).clip(lower=0.0)
        bloques.append(c)
    u = pd.concat(bloques, ignore_index=True)
    u["serie"] = u["SERIE"].astype(str).str.strip()
    u["emisora_v"] = u["EMISORA"].astype(str).str.strip()
    u["sucio"] = pd.to_numeric(u["PRECIO SUCIO"], errors="coerce")
    u["DURACION"] = pd.to_numeric(u["DURACION"], errors="coerce")
    return u.drop_duplicates(subset=["clase", "emisora_v", "serie"],
                             keep="first")


def valuar_cartera(dv: dict) -> pd.DataFrame:
    """
    Posiciones valuadas contra el vector con DV01 monetario. Los titulos
    por liquidar se suman a la exposicion economica. El reporto se valua al
    precio de registro (colateral a mercado) con DV01 cero. La columna
    dif_custodio_pct compara nuestra valuacion contra el Valor Mercado del
    estado del custodio cuando este viene en el registro.
    """
    pos = cargar_posiciones()
    if not len(pos):
        return pd.DataFrame()
    u = _universo(dv)
    m = pos.merge(
        u[["clase", "emisora_v", "serie", "anios", "ytm", "DURACION",
           "CONVEXIDAD", "sucio", "FECHA VCTO"]],
        left_on=["tipo_valor", "emisora", "serie"],
        right_on=["clase", "emisora_v", "serie"],
        how="left")
    es_repo = m["clase_op"].astype(str).str.upper().str.startswith("REPO")
    m["encontrado"] = m["sucio"].notna() | es_repo
    m["titulos_econ"] = m["titulos"] + m["por_liquidar"].fillna(0.0)
    m["valor"] = np.where(es_repo,
                          m["titulos"] * m["costo_unit"],
                          m["titulos_econ"] * m["sucio"])
    m["pnl"] = np.where(es_repo, 0.0, m["valor"] - m["costo_total"])
    m["dv01"] = np.where(
        es_repo, 0.0,
        pd.to_numeric(m["DURACION"], errors="coerce").fillna(0.0)
        * m["valor"] * 1e-4)
    m["es_repo"] = es_repo
    m["dif_custodio_pct"] = np.where(
        m["vm_ref"].notna() & (m["vm_ref"] != 0) & ~es_repo,
        (m["titulos"] * m["sucio"] / m["vm_ref"] - 1.0) * 100.0,
        float("nan"))
    total = m["valor"].sum()
    m["peso_pct"] = np.where(total > 0, m["valor"] / total * 100.0, 0.0)
    return m


def krd_por_cubeta(anios: float, dv01: float) -> dict[float, float]:
    """Reparte el DV01 entre las dos cubetas clave que flanquean el plazo."""
    t = max(float(anios), 0.01)
    if t <= CUBETAS[0]:
        return {CUBETAS[0]: dv01}
    if t >= CUBETAS[-1]:
        return {CUBETAS[-1]: dv01}
    for k1, k2 in zip(CUBETAS, CUBETAS[1:]):
        if k1 <= t <= k2:
            w2 = (t - k1) / (k2 - k1)
            return {k1: dv01 * (1 - w2), k2: dv01 * w2}
    return {}


def perfil_krd(df: pd.DataFrame, col_anios="anios",
               col_dv01="dv01") -> dict[float, float]:
    perfil = {k: 0.0 for k in CUBETAS}
    for r in df.itertuples():
        a = getattr(r, col_anios)
        d = getattr(r, col_dv01)
        if a == a and d == d:
            for k, v in krd_por_cubeta(a, d).items():
                perfil[k] += v
    return perfil


# --------------------------------------------------------------------------
# 3. Benchmark gubernamental y posicionamiento
# --------------------------------------------------------------------------

def benchmark_gubernamental(dv: dict) -> pd.DataFrame:
    """Bonos M ponderados por monto en circulacion (indice proxy)."""
    m = dv["bonos_m"].copy()
    m["circ"] = pd.to_numeric(m["MONTO EN CIRCULACION"], errors="coerce")
    m = m.dropna(subset=["circ"])
    m["peso"] = m["circ"] / m["circ"].sum()
    m["dv01_unit"] = m["DURACION"] * m["peso"]  # perfil por unidad invertida
    return m


def posicionamiento_krd(dv: dict, cartera: pd.DataFrame) -> pd.DataFrame:
    """
    DV01 por cubeta del portafolio (en % del DV01 total) contra el perfil
    del benchmark gubernamental: la foto de posicionamiento activo.
    """
    bench = benchmark_gubernamental(dv)
    p_bench = perfil_krd(bench.assign(dv01=bench["dv01_unit"]))
    tot_b = sum(p_bench.values()) or 1.0

    if len(cartera):
        sub = cartera[cartera["encontrado"]]
        p_port = perfil_krd(sub)
    else:
        p_port = {k: 0.0 for k in CUBETAS}
    tot_p = sum(p_port.values()) or 1.0

    filas = []
    for k in CUBETAS:
        pp = p_port[k] / tot_p * 100.0
        pb = p_bench[k] / tot_b * 100.0
        filas.append(dict(cubeta=f"{k:.0f}a", port_pct=pp, bench_pct=pb,
                          activo_pp=pp - pb,
                          port_mxn=p_port[k]))
    return pd.DataFrame(filas)


def te_ex_ante(hist: pd.DataFrame) -> float | None:
    """Se activa al acumular historia de vectores (>=20 sesiones)."""
    if hist is None or len(hist) < 20:
        return None
    dif = hist["m10"].diff().dropna()
    return float(dif.std(ddof=1) * np.sqrt(252))


# --------------------------------------------------------------------------
# 4. ALM real
# --------------------------------------------------------------------------

def cargar_pasivo() -> pd.DataFrame:
    if not CSV_PASIVO.exists():
        return pd.DataFrame()
    p = pd.read_csv(CSV_PASIVO)
    return p.dropna(subset=["anio", "flujo_real"])


def alm_real(dv: dict, cartera: pd.DataFrame) -> dict | None:
    """
    PV y KRD real del pasivo contra los udibonos en cartera, brecha por
    cubeta y cartera replicante sugerida (mínimos cuadrados, sin cortos).
    """
    pasivo = cargar_pasivo()
    if not len(pasivo):
        return None
    s = dv["udibonos"]
    an_s = s["anios"].values.astype(float)
    yt_s = s["ytm"].values.astype(float)

    def r_real(t):
        return float(np.interp(t, an_s, yt_s)) / 100.0

    # PV y KRD del pasivo (sensibilidad por cubeta via repartir el DV01 de
    # cada flujo, con duracion = plazo del flujo)
    pv = 0.0
    krd_pas = {k: 0.0 for k in CUBETAS}
    for r in pasivo.itertuples():
        t = float(r.anio)
        f = float(r.flujo_real)
        d = f / (1 + r_real(t)) ** t
        pv += d
        for k, v in krd_por_cubeta(t, d * t * 1e-4).items():
            krd_pas[k] += v
    dur_pasivo = sum(
        float(r.anio) * float(r.flujo_real) / (1 + r_real(float(r.anio))) ** float(r.anio)
        for r in pasivo.itertuples()) / pv

    # activo real: udibonos en cartera
    act = (cartera[(cartera["tipo_valor"] == "S") & cartera["encontrado"]]
           if len(cartera) else pd.DataFrame())
    krd_act = perfil_krd(act) if len(act) else {k: 0.0 for k in CUBETAS}
    valor_act = float(act["valor"].sum()) if len(act) else 0.0
    dur_act = (float((act["DURACION"] * act["valor"]).sum() / valor_act)
               if valor_act else 0.0)

    # cartera replicante: pesos no negativos de los 14 udibonos cuyas KRD
    # (por peso invertido) reproducen las del pasivo
    A = np.zeros((len(CUBETAS), len(s)))
    for j, rj in enumerate(s.itertuples()):
        for i, k in enumerate(CUBETAS):
            A[i, j] = krd_por_cubeta(float(rj.anios),
                                     float(rj.DURACION) * 1e-4).get(k, 0.0)
    b_vec = np.array([krd_pas[k] for k in CUBETAS])
    w, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    w = np.clip(w, 0.0, None)
    if w.sum() > 0:
        # re-escala para reproducir el DV01 real total del pasivo
        w *= b_vec.sum() / (A @ w).sum()
    replicante = pd.DataFrame(dict(
        serie=s["SERIE"].astype(str).values, anios=an_s,
        real=yt_s, inversion=w)).query("inversion > @pv * 0.001")

    filas = [dict(cubeta=f"{k:.0f}a", pasivo=krd_pas[k], activo=krd_act[k],
                  brecha=krd_act[k] - krd_pas[k]) for k in CUBETAS]
    return dict(pv_pasivo=pv, dur_pasivo=dur_pasivo,
                valor_activo=valor_act, dur_activo=dur_act,
                cubetas=pd.DataFrame(filas), replicante=replicante,
                dv01_pasivo=sum(krd_pas.values()),
                dv01_activo=sum(krd_act.values()))


# --------------------------------------------------------------------------
# 7. Constructor de estructuras DV01-neutral
# --------------------------------------------------------------------------

def _fila_m(dv: dict, serie: str):
    m = dv["bonos_m"]
    f = m[m["SERIE"].astype(str).str.strip() == serie]
    return f.iloc[0] if len(f) else None


def estructuras_dv01(dv: dict, base_titulos: float = 10_000.0) -> list[dict]:
    """
    Estructuras clasicas de mesa calibradas a DV01 neto ~cero, con la pata
    principal en `base_titulos` titulos. Carry 3m en MXN por pata:
    valor x (ytm - fondeo) x 0.25; el rolldown se toma de carry_rolldown.
    """
    cr = bn.carry_rolldown(dv)
    fondeo = float(cr["fondeo"].iloc[0])
    roll = dict(zip(cr["serie"], cr["roll_bp"]))
    res = bn.residuos_curva(dv)
    residuo = dict(zip(res["serie"], res["residuo_pb"]))

    def pata(serie, sentido, titulos):
        f = _fila_m(dv, serie)
        if f is None:
            return None
        sucio = float(f["PRECIO SUCIO"])
        valor = titulos * sucio
        dv01 = float(f["DURACION"]) * valor * 1e-4
        carry3m = valor * (float(f["ytm"]) - fondeo) / 100.0 * 0.25
        roll3m = valor * roll.get(serie, 0.0) / 10_000.0
        signo = 1.0 if sentido == "Compra" else -1.0
        return dict(serie=serie, sentido=sentido, titulos=titulos,
                    ytm=float(f["ytm"]), dv01=signo * dv01,
                    carry3m=signo * (carry3m + roll3m),
                    residuo=residuo.get(serie, 0.0))

    def calibrar(serie_obj, dv01_obj):
        f = _fila_m(dv, serie_obj)
        if f is None:
            return 0.0
        unit = float(f["DURACION"]) * float(f["PRECIO SUCIO"]) * 1e-4
        return dv01_obj / unit if unit else 0.0

    estructuras = []

    # a) Switch de valor relativo 361120 -> 381118
    p1 = pata("361120", "Venta", base_titulos)
    if p1:
        q2 = calibrar("381118", abs(p1["dv01"]))
        p2 = pata("381118", "Compra", q2)
        estructuras.append(dict(
            nombre="Switch RV: vender 361120 / comprar 381118",
            senal=(f"pickup {p2['ytm'] - p1['ytm']:+.2f} pp; vende papel "
                   f"{residuo.get('361120', 0):+.0f} pb caro y compra "
                   f"{residuo.get('381118', 0):+.0f} pb"),
            patas=[p1, p2]))

    # b) Mariposa: comprar panza 300228 vs 290301 y 360221
    pb_ = pata("300228", "Compra", base_titulos)
    if pb_:
        mitad = abs(pb_["dv01"]) / 2.0
        a1 = pata("290301", "Venta", calibrar("290301", mitad))
        a2 = pata("360221", "Venta", calibrar("360221", mitad))
        estructuras.append(dict(
            nombre="Mariposa: comprar 300228 vs 290301 y 360221",
            senal=(f"panza {residuo.get('300228', 0):+.0f} pb barata; "
                   f"fly 2s5s10s positivo"),
            patas=[pb_, a1, a2]))

    # c) Aplanador con carry: comprar 381118 / vender 280302
    c1 = pata("381118", "Compra", base_titulos)
    if c1:
        c2 = pata("280302", "Venta", calibrar("280302", abs(c1["dv01"])))
        estructuras.append(dict(
            nombre="Aplanador 1.6s12s: comprar 381118 / vender 280302",
            senal="apuesta a compresion de 2s10s cobrando el carry de la pendiente",
            patas=[c1, c2]))

    # cierre de cada estructura
    for e in estructuras:
        patas = [p for p in e["patas"] if p]
        e["patas"] = patas
        e["dv01_neto"] = sum(p["dv01"] for p in patas)
        e["carry3m_neto"] = sum(p["carry3m"] for p in patas)
        e["dv01_bruto"] = sum(abs(p["dv01"]) for p in patas)
    return estructuras
