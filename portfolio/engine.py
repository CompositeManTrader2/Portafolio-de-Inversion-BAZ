"""
Motor de posicion, efectivo y resultado.

Convenciones de costeo (las mismas que usa la mesa en el archivo fuente):

  COMPRA  ->  titulos += q
              costo_total += q * precio
              precio_costo = costo_total / titulos      (promedio ponderado)
              efectivo -= (q * precio + comision + IVA)

  VENTA   ->  titulos -= q
              precio_costo NO se modifica
              costo_total = titulos * precio_costo       (se reduce a prorrata)
              efectivo += (q * precio - comision - IVA)
              resultado realizado = q * (precio - precio_costo) - comision - IVA

Costos de operacion calibrados con la boleta del custodio (Res.104351):
comision = 4 pb sobre el monto bruto e IVA = 16 % sobre la comision.
Verificado: ALSEA 50,000 @ 41.44858 -> bruto 2,072,429.00, comision 828.9716,
IVA 132.635451, neto 2,073,390.607051.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .taxonomy import clasificar

# Costos de operacion por defecto (calibrados contra la boleta del custodio).
COMISION_BPS_DEFAULT = 4.0    # puntos base sobre el monto bruto
TASA_IVA_DEFAULT = 0.16       # IVA sobre la comision


@dataclass(frozen=True)
class CostosOperacion:
    comision_bps: float = COMISION_BPS_DEFAULT
    tasa_iva: float = TASA_IVA_DEFAULT

    def desglosar(self, bruto: float,
                  comision: float | None = None,
                  iva: float | None = None) -> tuple[float, float]:
        """
        Comision e IVA de una operacion. Si la boleta del custodio ya los
        trae, se respetan tal cual; si no, se estiman con la tarifa.
        """
        c = comision if comision is not None and not pd.isna(comision) \
            else abs(bruto) * self.comision_bps / 10_000.0
        i = iva if iva is not None and not pd.isna(iva) else c * self.tasa_iva
        return float(c), float(i)


@dataclass
class ResultadoMotor:
    """Salida completa del motor, lista para presentacion."""
    posiciones: pd.DataFrame          # cartera vigente
    bitacora: pd.DataFrame            # movimientos con su efecto en caja
    efectivo: float                   # saldo de efectivo final
    efectivo_inicial: float
    realizado: float                  # resultado realizado acumulado
    costos_totales: float             # comisiones + IVA pagados
    flujo_compras: float
    flujo_ventas: float
    incidencias: list[str] = field(default_factory=list)


def _ficha_taxonomica(emisora: str) -> dict:
    tax = clasificar(emisora)
    return {k: tax[k] for k in
            ("ticker", "clase_activo", "sector", "industria",
             "region", "mercado", "divisa_subyacente")}


def construir_posicion(base: pd.DataFrame,
                       movimientos: pd.DataFrame | None = None,
                       efectivo_inicial: float = 0.0,
                       costos: CostosOperacion | None = None,
                       hasta: date | None = None) -> ResultadoMotor:
    """
    Aplica los movimientos sobre la posicion base y devuelve la cartera
    vigente, el saldo de efectivo y el resultado realizado.

    `hasta` permite reconstruir la cartera a una fecha intermedia
    (los movimientos sin fecha siempre se aplican).
    """
    costos = costos or CostosOperacion()
    incidencias: list[str] = []

    # --- estado inicial ---------------------------------------------------
    estado: dict[str, dict] = {}
    for _, r in base.iterrows():
        emi = r["emisora"]
        estado[emi] = {
            "emisora": emi,
            "tipo_valor": r.get("tipo_valor", ""),
            "titulos": float(r["titulos"]),
            "precio_costo": float(r["precio_costo"]),
            "costo_total": float(r["titulos"]) * float(r["precio_costo"]),
            "precio_ref": r.get("precio_ref"),
            **_ficha_taxonomica(emi),
        }

    efectivo = float(efectivo_inicial)
    realizado = 0.0
    costos_totales = 0.0
    flujo_compras = 0.0
    flujo_ventas = 0.0
    bitacora: list[dict] = []

    movs = movimientos if movimientos is not None else pd.DataFrame()
    if len(movs) and hasta is not None:
        movs = movs[movs["fecha"].isna() | (movs["fecha"] <= hasta)]

    # --- aplicacion cronologica ------------------------------------------
    for _, m in movs.iterrows():
        emi = m["emisora"]
        op = m["operacion"]
        q = float(m["titulos"])
        p = float(m["precio"])
        bruto = q * p
        com, iva = costos.desglosar(bruto, m.get("comision"), m.get("iva"))

        pos = estado.get(emi)
        if pos is None:
            if op == "V":
                incidencias.append(
                    f"Venta de {emi} sin posicion previa; se registra en corto."
                )
            pos = {
                "emisora": emi, "tipo_valor": "",
                "titulos": 0.0, "precio_costo": 0.0, "costo_total": 0.0,
                "precio_ref": None, **_ficha_taxonomica(emi),
            }
            estado[emi] = pos

        costo_previo = pos["precio_costo"]
        titulos_previos = pos["titulos"]
        realizado_op = 0.0

        if op == "C":
            pos["titulos"] = titulos_previos + q
            pos["costo_total"] = pos["costo_total"] + bruto
            pos["precio_costo"] = (pos["costo_total"] / pos["titulos"]
                                   if pos["titulos"] else 0.0)
            efectivo -= (bruto + com + iva)
            flujo_compras += bruto
            efecto_caja = -(bruto + com + iva)
        else:  # venta
            if q > titulos_previos + 1e-9:
                incidencias.append(
                    f"Venta de {q:,.0f} titulos de {emi} excede los "
                    f"{titulos_previos:,.0f} en posicion."
                )
            pos["titulos"] = titulos_previos - q
            # El precio de costo no se toca en ventas; el costo total baja a prorrata.
            pos["costo_total"] = pos["titulos"] * pos["precio_costo"]
            realizado_op = q * (p - costo_previo) - com - iva
            realizado += realizado_op
            efectivo += (bruto - com - iva)
            flujo_ventas += bruto
            efecto_caja = bruto - com - iva

        costos_totales += com + iva

        bitacora.append({
            "fecha": m.get("fecha"),
            "operacion": "Compra" if op == "C" else "Venta",
            "emisora": emi,
            "titulos": q,
            "precio": p,
            "monto_bruto": bruto,
            "comision": com,
            "iva": iva,
            "monto_neto": bruto + com + iva if op == "C" else bruto - com - iva,
            "efecto_caja": efecto_caja,
            "costo_promedio_previo": costo_previo,
            "resultado_realizado": realizado_op,
            "titulos_despues": pos["titulos"],
            "costo_promedio_despues": pos["precio_costo"],
            "efectivo_despues": efectivo,
            "fuente": m.get("fuente", ""),
        })

    # --- cartera vigente --------------------------------------------------
    vivas = [p for p in estado.values() if abs(p["titulos"]) > 1e-9]
    posiciones = pd.DataFrame(vivas) if vivas else pd.DataFrame(
        columns=["emisora", "titulos", "precio_costo", "costo_total"]
    )
    if len(posiciones):
        posiciones = posiciones.sort_values(
            "costo_total", ascending=False).reset_index(drop=True)

    bit = pd.DataFrame(bitacora)
    if len(bit):
        bit = bit.reset_index(drop=True)

    return ResultadoMotor(
        posiciones=posiciones,
        bitacora=bit,
        efectivo=efectivo,
        efectivo_inicial=float(efectivo_inicial),
        realizado=realizado,
        costos_totales=costos_totales,
        flujo_compras=flujo_compras,
        flujo_ventas=flujo_ventas,
        incidencias=incidencias,
    )


def valuar(posiciones: pd.DataFrame,
           precios: dict[str, float],
           precios_previos: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Agrega valuacion a mercado sobre la cartera vigente.

    `precios` mapea ticker -> ultimo precio; `precios_previos` mapea
    ticker -> cierre anterior, para el resultado del dia.
    """
    if not len(posiciones):
        return posiciones.copy()

    df = posiciones.copy()
    precios_previos = precios_previos or {}

    df["precio_mercado"] = df["ticker"].map(precios)
    df["cierre_previo"] = df["ticker"].map(precios_previos)

    # Sin precio vivo se cae al precio de referencia del archivo, y en
    # ultima instancia al costo, para no romper la valuacion total.
    faltantes = df["precio_mercado"].isna()
    if faltantes.any():
        df.loc[faltantes, "precio_mercado"] = (
            df.loc[faltantes, "precio_ref"].fillna(
                df.loc[faltantes, "precio_costo"])
        )
    df["precio_estimado"] = faltantes

    df["valor_mercado"] = df["titulos"] * df["precio_mercado"]
    df["no_realizado"] = df["valor_mercado"] - df["costo_total"]
    df["rend_pct"] = np.where(
        df["costo_total"] != 0,
        df["no_realizado"] / df["costo_total"] * 100.0,
        0.0,
    )

    df["pnl_dia"] = np.where(
        df["cierre_previo"].notna(),
        df["titulos"] * (df["precio_mercado"] - df["cierre_previo"]),
        0.0,
    )
    df["var_dia_pct"] = np.where(
        df["cierre_previo"].notna() & (df["cierre_previo"] != 0),
        (df["precio_mercado"] / df["cierre_previo"] - 1.0) * 100.0,
        np.nan,
    )

    total = df["valor_mercado"].sum()
    df["peso_pct"] = df["valor_mercado"] / total * 100.0 if total else 0.0

    return df.sort_values("valor_mercado", ascending=False).reset_index(drop=True)


def resumen_cartera(valuada: pd.DataFrame, efectivo: float,
                    realizado: float) -> dict:
    """Cifras de cabecera del portafolio."""
    if not len(valuada):
        return dict(valor_instrumentos=0.0, efectivo=efectivo,
                    valor_total=efectivo, costo_total=0.0, no_realizado=0.0,
                    realizado=realizado, pnl_total=realizado,
                    rend_pct=0.0, pnl_dia=0.0, var_dia_pct=0.0,
                    n_posiciones=0, peso_efectivo_pct=100.0 if efectivo else 0.0)

    valor_inst = float(valuada["valor_mercado"].sum())
    costo = float(valuada["costo_total"].sum())
    no_real = float(valuada["no_realizado"].sum())
    pnl_dia = float(valuada["pnl_dia"].sum())
    valor_total = valor_inst + efectivo
    base_dia = valor_inst - pnl_dia

    return dict(
        valor_instrumentos=valor_inst,
        efectivo=efectivo,
        valor_total=valor_total,
        costo_total=costo,
        no_realizado=no_real,
        realizado=realizado,
        pnl_total=no_real + realizado,
        rend_pct=(no_real / costo * 100.0) if costo else 0.0,
        pnl_dia=pnl_dia,
        var_dia_pct=(pnl_dia / base_dia * 100.0) if base_dia else 0.0,
        n_posiciones=int(len(valuada)),
        peso_efectivo_pct=(efectivo / valor_total * 100.0) if valor_total else 0.0,
    )


def serie_efectivo(bitacora: pd.DataFrame, efectivo_inicial: float,
                   fecha_inicial: date | None = None) -> pd.DataFrame:
    """
    Evolucion diaria del saldo de liquidez a partir de la bitacora.

    Con `fecha_inicial` la serie arranca con el saldo de apertura en la fecha
    de la posicion base; sin ella, el primer punto seria ya el saldo posterior
    al primer dia operado y el nivel de partida no se veria en la grafica.
    """
    if not len(bitacora) or bitacora["fecha"].isna().all():
        return pd.DataFrame({"fecha": [], "efectivo": [], "flujo": []})

    b = bitacora.dropna(subset=["fecha"]).copy()
    diario = b.groupby("fecha", as_index=False)["efecto_caja"].sum()
    diario = diario.rename(columns={"efecto_caja": "flujo"}).sort_values("fecha")

    if fecha_inicial is not None and fecha_inicial < diario["fecha"].min():
        diario = pd.concat([
            pd.DataFrame([{"fecha": fecha_inicial, "flujo": 0.0}]),
            diario,
        ], ignore_index=True)

    diario["efectivo"] = efectivo_inicial + diario["flujo"].cumsum()
    return diario.reset_index(drop=True)


def agrupar(valuada: pd.DataFrame, dimension: str,
            efectivo: float = 0.0,
            incluir_efectivo: bool = False) -> pd.DataFrame:
    """
    Agrega la cartera por una dimension (sector, region, clase_activo, ...)
    devolviendo valor, costo, P&L y peso.
    """
    if not len(valuada):
        return pd.DataFrame(columns=[dimension, "valor_mercado", "costo_total",
                                     "no_realizado", "pnl_dia", "peso_pct",
                                     "rend_pct"])

    g = (valuada.groupby(dimension, as_index=False)
         .agg(valor_mercado=("valor_mercado", "sum"),
              costo_total=("costo_total", "sum"),
              no_realizado=("no_realizado", "sum"),
              pnl_dia=("pnl_dia", "sum"),
              n_posiciones=("emisora", "count")))

    if incluir_efectivo and efectivo:
        g = pd.concat([g, pd.DataFrame([{
            dimension: "Efectivo", "valor_mercado": efectivo,
            "costo_total": efectivo, "no_realizado": 0.0,
            "pnl_dia": 0.0, "n_posiciones": 0,
        }])], ignore_index=True)

    total = g["valor_mercado"].sum()
    g["peso_pct"] = g["valor_mercado"] / total * 100.0 if total else 0.0
    g["rend_pct"] = np.where(g["costo_total"] != 0,
                             g["no_realizado"] / g["costo_total"] * 100.0, 0.0)
    return g.sort_values("valor_mercado", ascending=False).reset_index(drop=True)
