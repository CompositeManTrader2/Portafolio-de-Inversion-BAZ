"""
Analitica de portafolio: riesgo, atribucion, concentracion y diagnostico.

El objetivo de este modulo es responder dos preguntas de mesa:
  - Que esta aportando y que esta restando (atribucion).
  - Cuanto riesgo estoy corriendo y de donde viene (descomposicion de riesgo).

Todas las series de riesgo se calculan sobre rendimientos logaritmicos
diarios y se anualizan con 252 sesiones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESIONES_ANIO = 252


# --------------------------------------------------------------------------
# Serie de valor del portafolio
# --------------------------------------------------------------------------

def serie_valor_portafolio(historico: pd.DataFrame,
                           posiciones: pd.DataFrame) -> pd.Series:
    """
    Valor de mercado historico de la cartera VIGENTE.

    Es una serie a titulos constantes: aplica la tenencia actual sobre los
    precios historicos. Sirve para medir el riesgo del portafolio tal como
    esta hoy, no para reconstruir el desempeno realizado del pasado.

    Solo entran las emisoras con historico suficiente y solo las fechas en
    que TODAS ellas cotizaron. Sumar columnas con huecos daria un salto
    artificial el dia que una emisora esporadica aparece por primera vez,
    que se leeria como un movimiento de mercado inexistente.
    """
    if not len(posiciones) or not len(historico):
        return pd.Series(dtype=float)

    utilizables, _ = cobertura_historica(historico, posiciones)
    titulos = posiciones.set_index("ticker")["titulos"]
    comunes = [t for t in utilizables if t in titulos.index]
    if not comunes:
        return pd.Series(dtype=float)

    precios = historico[comunes].ffill().dropna()
    if not len(precios):
        return pd.Series(dtype=float)
    return (precios * titulos.reindex(comunes)).sum(axis=1)


def rendimientos_portafolio(historico: pd.DataFrame,
                            posiciones: pd.DataFrame) -> pd.Series:
    valor = serie_valor_portafolio(historico, posiciones)
    if len(valor) < 2:
        return pd.Series(dtype=float)
    return np.log(valor / valor.shift(1)).dropna()


# --------------------------------------------------------------------------
# Metricas de riesgo
# --------------------------------------------------------------------------

def metricas_riesgo(rend_port: pd.Series,
                    rend_bench: pd.Series | None = None,
                    tasa_libre_anual: float = 0.09) -> dict:
    """
    Bloque de riesgo estandar. `tasa_libre_anual` por defecto ~ Cetes 28d,
    que es la referencia correcta para un mandato en pesos.
    """
    vacio = dict(vol_anual=np.nan, rend_anual=np.nan, sharpe=np.nan,
                 sortino=np.nan, max_drawdown=np.nan, var_95=np.nan,
                 var_99=np.nan, cvar_95=np.nan, beta=np.nan, alpha_anual=np.nan,
                 correlacion=np.nan, tracking_error=np.nan, info_ratio=np.nan,
                 sesiones=0)
    if rend_port is None or len(rend_port) < 20:
        return vacio

    r = rend_port.dropna()
    vol = float(r.std(ddof=1) * np.sqrt(SESIONES_ANIO))
    # La media de rendimientos log anualizada se convierte a rendimiento
    # efectivo (exp - 1) antes de compararla con la tasa libre, que es
    # aritmetica. Restar una log-media de una tasa efectiva sesga el Sharpe
    # a la baja, mas cuanto mayor la volatilidad.
    rend_anual = float(np.expm1(r.mean() * SESIONES_ANIO))
    rf = tasa_libre_anual

    bajistas = r[r < 0]
    vol_bajista = (float(bajistas.std(ddof=1) * np.sqrt(SESIONES_ANIO))
                   if len(bajistas) > 1 else np.nan)

    acumulado = np.exp(r.cumsum())
    max_dd = float((acumulado / acumulado.cummax() - 1.0).min() * 100.0)

    m = dict(
        vol_anual=vol * 100.0,
        rend_anual=rend_anual * 100.0,
        sharpe=(rend_anual - rf) / vol if vol else np.nan,
        sortino=((rend_anual - rf) / vol_bajista
                 if vol_bajista and vol_bajista == vol_bajista else np.nan),
        max_drawdown=max_dd,
        # VaR historico: percentil empirico del rendimiento diario.
        var_95=float(np.percentile(r, 5) * 100.0),
        var_99=float(np.percentile(r, 1) * 100.0),
        cvar_95=float(r[r <= np.percentile(r, 5)].mean() * 100.0),
        beta=np.nan, alpha_anual=np.nan, correlacion=np.nan,
        tracking_error=np.nan, info_ratio=np.nan,
        sesiones=int(len(r)),
    )

    if rend_bench is not None and len(rend_bench) >= 20:
        par = pd.concat([r, rend_bench], axis=1, join="inner").dropna()
        par.columns = ["port", "bench"]
        if len(par) >= 20 and par["bench"].std() > 0:
            cov = float(par.cov().iloc[0, 1])
            var_b = float(par["bench"].var(ddof=1))
            beta = cov / var_b if var_b else np.nan
            rend_b_anual = float(np.expm1(par["bench"].mean() * SESIONES_ANIO))
            activo = par["port"] - par["bench"]
            te = float(activo.std(ddof=1) * np.sqrt(SESIONES_ANIO))
            m.update(
                beta=beta,
                alpha_anual=(rend_anual - (rf + beta * (rend_b_anual - rf))) * 100.0,
                correlacion=float(par.corr().iloc[0, 1]),
                tracking_error=te * 100.0,
                info_ratio=(float(activo.mean() * SESIONES_ANIO) / te) if te else np.nan,
            )
    return m


def descomposicion_riesgo(historico: pd.DataFrame,
                          valuada: pd.DataFrame) -> pd.DataFrame:
    """
    Contribucion de cada posicion al riesgo total (metodologia de
    contribucion marginal): CTR_i = w_i * (Sigma w)_i / sigma_p.

    Las contribuciones suman exactamente la volatilidad del portafolio, lo
    que permite ver que posicion realmente esta consumiendo el presupuesto
    de riesgo, no solo cual pesa mas.
    """
    if not len(valuada) or not len(historico):
        return pd.DataFrame()

    activos = [t for t in valuada["ticker"] if t in historico.columns]
    if len(activos) < 2:
        return pd.DataFrame()

    rend = np.log(historico[activos] / historico[activos].shift(1)).dropna(how="all")
    rend = rend.dropna(axis=1, thresh=int(len(rend) * 0.6)).dropna()
    activos = list(rend.columns)
    if len(activos) < 2 or len(rend) < 30:
        return pd.DataFrame()

    sub = valuada[valuada["ticker"].isin(activos)].copy()
    pesos = (sub.set_index("ticker")["valor_mercado"].reindex(activos))
    pesos = pesos / pesos.sum()

    cov = rend.cov().values * SESIONES_ANIO
    w = pesos.values
    var_p = float(w @ cov @ w)
    if var_p <= 0:
        return pd.DataFrame()
    vol_p = np.sqrt(var_p)

    contrib_marginal = (cov @ w) / vol_p          # dSigma/dw_i
    contrib = w * contrib_marginal                # suman vol_p

    fuera = rend.std(ddof=1).values * np.sqrt(SESIONES_ANIO)

    out = pd.DataFrame({
        "ticker": activos,
        "peso_pct": w * 100.0,
        "vol_individual_pct": fuera * 100.0,
        "contrib_riesgo_pct": contrib / vol_p * 100.0,   # % del riesgo total
        "contrib_riesgo_abs": contrib * 100.0,           # en puntos de vol anual
    })
    out = out.merge(valuada[["ticker", "emisora", "sector", "region"]],
                    on="ticker", how="left")
    # Ratio > 1 => la posicion aporta mas riesgo que peso.
    out["riesgo_sobre_peso"] = out["contrib_riesgo_pct"] / out["peso_pct"].replace(0, np.nan)
    return out.sort_values("contrib_riesgo_pct", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Concentracion
# --------------------------------------------------------------------------

def metricas_concentracion(valuada: pd.DataFrame) -> dict:
    """Herfindahl, numero efectivo de posiciones y pesos acumulados."""
    if not len(valuada):
        return dict(hhi=np.nan, n_efectivo=np.nan, top5_pct=np.nan,
                    top10_pct=np.nan, mayor_pct=np.nan, mayor_nombre="")

    w = (valuada["valor_mercado"] / valuada["valor_mercado"].sum()).sort_values(
        ascending=False)
    hhi = float((w ** 2).sum())
    return dict(
        hhi=hhi * 10_000,                 # escala 0-10,000
        n_efectivo=1.0 / hhi if hhi else np.nan,
        top5_pct=float(w.head(5).sum() * 100.0),
        top10_pct=float(w.head(10).sum() * 100.0),
        mayor_pct=float(w.iloc[0] * 100.0),
        mayor_nombre=str(valuada.sort_values("valor_mercado",
                                             ascending=False)["emisora"].iloc[0]),
    )


# --------------------------------------------------------------------------
# Atribucion
# --------------------------------------------------------------------------

def atribucion(valuada: pd.DataFrame, dimension: str = "sector") -> pd.DataFrame:
    """
    Contribucion al resultado no realizado por dimension.

    contribucion_pct = P&L del grupo / costo total de la cartera, de modo que
    la suma de las contribuciones reproduce el rendimiento del portafolio.
    """
    if not len(valuada):
        return pd.DataFrame()

    costo_total = float(valuada["costo_total"].sum())
    g = (valuada.groupby(dimension, as_index=False)
         .agg(valor_mercado=("valor_mercado", "sum"),
              costo_total=("costo_total", "sum"),
              no_realizado=("no_realizado", "sum"),
              pnl_dia=("pnl_dia", "sum"),
              n=("emisora", "count")))
    g["rend_pct"] = np.where(g["costo_total"] != 0,
                             g["no_realizado"] / g["costo_total"] * 100.0, 0.0)
    g["peso_pct"] = g["valor_mercado"] / valuada["valor_mercado"].sum() * 100.0
    g["contrib_pct"] = (g["no_realizado"] / costo_total * 100.0
                        if costo_total else 0.0)
    return g.sort_values("contrib_pct", ascending=False).reset_index(drop=True)


def ganadores_perdedores(valuada: pd.DataFrame, n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Las n posiciones que mas suman y las n que mas restan, en pesos."""
    if not len(valuada):
        vacio = pd.DataFrame()
        return vacio, vacio
    cols = ["emisora", "sector", "region", "titulos", "precio_costo",
            "precio_mercado", "valor_mercado", "no_realizado", "rend_pct",
            "peso_pct", "pnl_dia"]
    cols = [c for c in cols if c in valuada.columns]
    orden = valuada.sort_values("no_realizado", ascending=False)
    return (orden.head(n)[cols].reset_index(drop=True),
            orden.tail(n)[cols].iloc[::-1].reset_index(drop=True))


# --------------------------------------------------------------------------
# Exposicion cambiaria
# --------------------------------------------------------------------------

def exposicion_fx(valuada: pd.DataFrame) -> dict:
    """
    Exposicion economica al dolar. Las emisoras del SIC cotizan en pesos pero
    su valor subyacente esta en dolares, por lo que el portafolio tiene una
    posicion larga en USDMXN implicita.
    """
    if not len(valuada):
        return dict(usd_pct=0.0, usd_monto=0.0, mxn_monto=0.0,
                    impacto_1pct_fx=0.0)
    total = float(valuada["valor_mercado"].sum())
    usd = float(valuada.loc[valuada["divisa_subyacente"] == "USD",
                            "valor_mercado"].sum())
    return dict(
        usd_pct=usd / total * 100.0 if total else 0.0,
        usd_monto=usd,
        mxn_monto=total - usd,
        # Un movimiento de 1 % en USDMXN mueve el valor en pesos de la
        # porcion dolarizada en la misma proporcion.
        impacto_1pct_fx=usd * 0.01,
    )


# --------------------------------------------------------------------------
# Diagnostico automatico
# --------------------------------------------------------------------------

def diagnostico(valuada: pd.DataFrame, riesgo: dict, concentracion: dict,
                tecnicos: pd.DataFrame | None = None,
                riesgo_pos: pd.DataFrame | None = None,
                peso_efectivo_pct: float = 0.0) -> list[dict]:
    """
    Reglas de mesa que traducen las metricas en observaciones accionables.

    Cada hallazgo trae severidad ('critico' | 'atencion' | 'favorable') para
    que la interfaz lo pinte con la escala de estado, nunca solo con color.
    """
    hallazgos: list[dict] = []

    def añadir(sev, titulo, detalle):
        hallazgos.append(dict(severidad=sev, titulo=titulo, detalle=detalle))

    # -- Concentracion --
    if concentracion.get("mayor_pct", 0) > 12:
        añadir("atencion", "Concentracion en una sola emisora",
               f"{concentracion['mayor_nombre']} representa "
               f"{concentracion['mayor_pct']:.1f} % del portafolio. Por encima de "
               f"10-12 % una sola posicion domina el resultado del mandato.")
    if concentracion.get("top5_pct", 0) > 55:
        añadir("atencion", "Cartera dominada por el top 5",
               f"Las cinco mayores posiciones concentran "
               f"{concentracion['top5_pct']:.1f} % del valor. La diversificacion "
               f"efectiva es de solo {concentracion.get('n_efectivo', 0):.1f} posiciones.")
    elif concentracion.get("n_efectivo", 0) >= 12:
        añadir("favorable", "Diversificacion efectiva adecuada",
               f"El numero efectivo de posiciones es "
               f"{concentracion['n_efectivo']:.1f}, consistente con una cartera "
               f"bien repartida.")

    # -- Riesgo --
    if riesgo.get("max_drawdown") == riesgo.get("max_drawdown") and \
            riesgo["max_drawdown"] < -18:
        añadir("critico", "Caida maxima elevada",
               f"La cartera actual habria sufrido una caida maxima de "
               f"{riesgo['max_drawdown']:.1f} % en la ventana analizada.")
    if riesgo.get("sharpe") == riesgo.get("sharpe"):
        if riesgo["sharpe"] < 0:
            añadir("critico", "Sharpe negativo",
                   f"Sharpe de {riesgo['sharpe']:.2f}: el portafolio no esta "
                   f"compensando el riesgo frente a la tasa libre de riesgo.")
        elif riesgo["sharpe"] > 0.8:
            añadir("favorable", "Relacion riesgo-rendimiento solida",
                   f"Sharpe de {riesgo['sharpe']:.2f} sobre la ventana analizada.")
    if riesgo.get("beta") == riesgo.get("beta") and riesgo["beta"] > 1.15:
        añadir("atencion", "Beta por encima del mercado",
               f"Beta de {riesgo['beta']:.2f} contra el IPC: la cartera amplifica "
               f"los movimientos del mercado en ambos sentidos.")
    if riesgo.get("info_ratio") == riesgo.get("info_ratio") and \
            riesgo["info_ratio"] < -0.3:
        añadir("atencion", "Gestion activa restando valor",
               f"Information ratio de {riesgo['info_ratio']:.2f} con tracking "
               f"error de {riesgo.get('tracking_error', 0):.1f} %: se esta "
               f"corriendo riesgo activo sin compensacion.")

    # -- Riesgo concentrado en pocas posiciones --
    if riesgo_pos is not None and len(riesgo_pos):
        top = riesgo_pos.head(3)
        cuota = float(top["contrib_riesgo_pct"].sum())
        if cuota > 45:
            añadir("atencion", "Presupuesto de riesgo concentrado",
                   f"{', '.join(top['emisora'].astype(str))} explican "
                   f"{cuota:.0f} % del riesgo total del portafolio.")
        desalineadas = riesgo_pos[riesgo_pos["riesgo_sobre_peso"] > 1.8]
        if len(desalineadas):
            nombres = ", ".join(desalineadas.head(3)["emisora"].astype(str))
            añadir("atencion", "Posiciones que aportan mas riesgo que peso",
                   f"{nombres} contribuyen al riesgo mas del doble de lo que "
                   f"pesan en la cartera.")

    # -- Posiciones perdedoras relevantes --
    if len(valuada):
        malas = valuada[(valuada["rend_pct"] < -15) & (valuada["peso_pct"] > 1.0)]
        if len(malas):
            detalle = "; ".join(
                f"{r.emisora} {r.rend_pct:+.1f} % ({r.peso_pct:.1f} % de la cartera)"
                for r in malas.head(4).itertuples())
            añadir("critico", "Posiciones con deterioro material", detalle)

        buenas = valuada[(valuada["rend_pct"] > 15) & (valuada["peso_pct"] > 1.0)]
        if len(buenas):
            detalle = "; ".join(
                f"{r.emisora} {r.rend_pct:+.1f} %"
                for r in buenas.head(4).itertuples())
            añadir("favorable", "Posiciones aportando de forma consistente", detalle)

    # -- Efectivo --
    if peso_efectivo_pct < -1:
        añadir("critico", "Saldo de efectivo negativo",
               f"El efectivo representa {peso_efectivo_pct:.1f} % del portafolio: "
               f"las compras superan a las ventas y el saldo esta sobregirado. "
               f"Verifica el efectivo inicial en el panel lateral.")
    elif peso_efectivo_pct > 12:
        añadir("atencion", "Efectivo ocioso elevado",
               f"El efectivo pesa {peso_efectivo_pct:.1f} % del portafolio. "
               f"Fuera de una postura defensiva deliberada, representa un lastre "
               f"sobre el rendimiento.")

    # -- Senales tecnicas --
    if tecnicos is not None and len(tecnicos):
        t = tecnicos.merge(valuada[["ticker", "emisora", "peso_pct"]],
                           on="ticker", how="inner")
        debiles = t[(t["vs_media200_pct"] < -10) & (t["peso_pct"] > 1.5)]
        if len(debiles):
            añadir("atencion", "Emisoras en tendencia bajista",
                   f"{', '.join(debiles.head(4)['emisora'].astype(str))} cotizan "
                   f"mas de 10 % por debajo de su media de 200 sesiones.")
        sobrecompra = t[(t["rsi14"] > 72) & (t["peso_pct"] > 1.5)]
        if len(sobrecompra):
            añadir("atencion", "Zona de sobrecompra",
                   f"{', '.join(sobrecompra.head(4)['emisora'].astype(str))} "
                   f"presentan RSI por arriba de 72; conviene evaluar toma de utilidad.")
        sobreventa = t[(t["rsi14"] < 30) & (t["peso_pct"] > 1.5)]
        if len(sobreventa):
            añadir("favorable", "Zona de sobreventa",
                   f"{', '.join(sobreventa.head(4)['emisora'].astype(str))} "
                   f"presentan RSI por debajo de 30; posible punto de acumulacion.")

    orden = {"critico": 0, "atencion": 1, "favorable": 2}
    return sorted(hallazgos, key=lambda h: orden.get(h["severidad"], 9))


def cobertura_historica(historico: pd.DataFrame, valuada: pd.DataFrame,
                        minimo: float = 0.6) -> tuple[list[str], list[str]]:
    """
    Separa las posiciones con historico utilizable de las que no lo tienen.

    Algunas emisoras del SIC cotizan de forma muy esporadica y traen precio
    vigente pero apenas un punado de observaciones historicas. Se valuan sin
    problema, pero no pueden entrar en correlaciones ni en el calculo de
    riesgo, y conviene decir cuales quedaron fuera en vez de omitirlas en
    silencio.
    """
    if not len(valuada) or not len(historico):
        return [], []
    umbral = len(historico) * minimo
    con, sin = [], []
    for t, emisora in zip(valuada["ticker"], valuada["emisora"]):
        if t in historico.columns and historico[t].notna().sum() >= umbral:
            con.append(t)
        else:
            sin.append(str(emisora))
    return con, sin


def matriz_correlacion(historico: pd.DataFrame, valuada: pd.DataFrame,
                       top: int = 15) -> pd.DataFrame:
    """Correlacion de rendimientos diarios entre las mayores posiciones."""
    if not len(valuada) or not len(historico):
        return pd.DataFrame()

    # Se descartan primero las columnas con historico insuficiente: si se
    # aplicara dropna por filas con una sola emisora esporadica presente,
    # esa emisora vaciaria toda la matriz.
    utilizables, _ = cobertura_historica(historico, valuada)
    princ = valuada[valuada["ticker"].isin(utilizables)].head(top)
    activos = [t for t in princ["ticker"] if t in historico.columns]
    if len(activos) < 2:
        return pd.DataFrame()

    rend = np.log(historico[activos] / historico[activos].shift(1)).dropna()
    if len(rend) < 30:
        return pd.DataFrame()
    corr = rend.corr()
    etiquetas = dict(zip(princ["ticker"], princ["emisora"]))
    corr.index = [etiquetas.get(t, t) for t in corr.index]
    corr.columns = [etiquetas.get(t, t) for t in corr.columns]
    return corr


# --------------------------------------------------------------------------
# Atribucion Brinson-Fachler
# --------------------------------------------------------------------------

def brinson_fachler(valuada: pd.DataFrame, historico: pd.DataFrame,
                    bench_sectores: pd.DataFrame,
                    sesiones: int) -> dict:
    """
    Atribucion del retorno activo contra el benchmark, por sector, con la
    descomposicion de Brinson-Fachler:

      Asignacion_i  = (w_p_i - w_b_i) x (r_b_i - r_B)
      Seleccion_i   =  w_b_i x (r_p_i - r_b_i)
      Interaccion_i = (w_p_i - w_b_i) x (r_p_i - r_b_i)

    donde r_B es el retorno del benchmark reconstruido. Con esa definicion la
    identidad  suma de efectos = r_p - r_B  cierra de forma exacta.

    Convenciones para casos frontera:
      - Sector del portafolio ausente del benchmark (posiciones SIC/US):
        apuesta fuera de indice -> todo el efecto es asignacion,
        w_p_i x (r_p_i - r_B). Seleccion e interaccion son cero.
      - Sector del benchmark no tenido: asignacion -w_b_i x (r_b_i - r_B).

    Limitaciones declaradas: pesos del portafolio al cierre (no promedio del
    periodo) y rendimientos a titulos constantes. Ambas se documentan en la
    interfaz; con pocos dias de historia de operaciones son de segundo orden.
    """
    if not len(valuada) or not len(historico) or not len(bench_sectores):
        return dict(tabla=pd.DataFrame(), r_p=np.nan, r_b=np.nan,
                    activo=np.nan, asignacion=np.nan, seleccion=np.nan,
                    interaccion=np.nan, sesiones=0)

    # --- rendimiento del periodo por posicion del portafolio --------------
    rends = {}
    for _, fila in valuada.iterrows():
        t = fila["ticker"]
        if t not in historico.columns:
            continue
        s = historico[t].dropna().tail(sesiones + 1)
        if len(s) >= 2 and s.iloc[0]:
            rends[t] = float(s.iloc[-1] / s.iloc[0] - 1.0)

    port = valuada[valuada["ticker"].isin(rends)].copy()
    if not len(port):
        return dict(tabla=pd.DataFrame(), r_p=np.nan, r_b=np.nan,
                    activo=np.nan, asignacion=np.nan, seleccion=np.nan,
                    interaccion=np.nan, sesiones=0)
    port["r"] = port["ticker"].map(rends)
    port["w"] = port["valor_mercado"] / port["valor_mercado"].sum()

    p_sec = (port.groupby("sector")
             .apply(lambda g: pd.Series({
                 "w_p": g["w"].sum(),
                 "r_p": float((g["w"] * g["r"]).sum() / g["w"].sum()),
             }), include_groups=False)
             .reset_index())

    b_sec = bench_sectores.rename(columns={"peso_pct": "w_b",
                                           "rendimiento_pct": "r_b"}).copy()
    b_sec["w_b"] = b_sec["w_b"] / 100.0
    b_sec["r_b"] = b_sec["r_b"] / 100.0

    m = p_sec.merge(b_sec[["sector", "w_b", "r_b"]], on="sector", how="outer")
    m[["w_p", "w_b"]] = m[["w_p", "w_b"]].fillna(0.0)

    r_B = float((m["w_b"] * m["r_b"].fillna(0.0)).sum())
    r_p = float((m["w_p"] * m["r_p"].fillna(0.0)).sum())

    filas = []
    for _, x in m.iterrows():
        w_p, w_b = float(x["w_p"]), float(x["w_b"])
        r_p_i = x["r_p"] if x["r_p"] == x["r_p"] else None
        r_b_i = x["r_b"] if x["r_b"] == x["r_b"] else None
        fuera = r_b_i is None or w_b == 0.0

        if fuera and r_p_i is not None:
            asig, sel, inter = w_p * (r_p_i - r_B), 0.0, 0.0
        elif r_p_i is None:                 # sector del indice no tenido
            asig, sel, inter = -w_b * (r_b_i - r_B), 0.0, 0.0
        else:
            asig = (w_p - w_b) * (r_b_i - r_B)
            sel = w_b * (r_p_i - r_b_i)
            inter = (w_p - w_b) * (r_p_i - r_b_i)

        filas.append(dict(
            sector=x["sector"], fuera_indice=fuera and r_p_i is not None,
            w_p_pct=w_p * 100.0, w_b_pct=w_b * 100.0,
            r_p_pct=(r_p_i * 100.0) if r_p_i is not None else np.nan,
            r_b_pct=(r_b_i * 100.0) if r_b_i is not None else np.nan,
            asignacion_pp=asig * 100.0, seleccion_pp=sel * 100.0,
            interaccion_pp=inter * 100.0,
            total_pp=(asig + sel + inter) * 100.0,
        ))

    tabla = (pd.DataFrame(filas)
             .sort_values("total_pp", key=lambda s: s.abs(), ascending=False)
             .reset_index(drop=True))

    return dict(
        tabla=tabla,
        r_p=r_p * 100.0, r_b=r_B * 100.0, activo=(r_p - r_B) * 100.0,
        asignacion=float(tabla["asignacion_pp"].sum()),
        seleccion=float(tabla["seleccion_pp"].sum()),
        interaccion=float(tabla["interaccion_pp"].sum()),
        sesiones=sesiones,
    )


# --------------------------------------------------------------------------
# Metricas extendidas de riesgo y rendimiento
# --------------------------------------------------------------------------

def metricas_extendidas(rend_port: pd.Series, rend_bench: pd.Series | None,
                        tasa_libre_anual: float = 0.09,
                        beta: float = np.nan,
                        vol_bench_anual: float | None = None) -> dict:
    """
    Segundo bloque de metricas institucionales sobre rendimientos diarios:
    Treynor, Calmar, M2, capturas alcista/bajista, beta bajista, hit ratio,
    mejor/peor dia, asimetria, curtosis y VaR parametrico.
    """
    vacio = dict(treynor=np.nan, calmar=np.nan, m2=np.nan,
                 captura_alcista=np.nan, captura_bajista=np.nan,
                 beta_bajista=np.nan, hit_ratio=np.nan,
                 mejor_dia=np.nan, peor_dia=np.nan,
                 asimetria=np.nan, curtosis=np.nan, var_param_95=np.nan)
    if rend_port is None or len(rend_port) < 20:
        return vacio

    rl = rend_port.dropna()
    r = np.expm1(rl)                        # simples para capturas y ratios
    rf = tasa_libre_anual
    rend_anual = float(np.expm1(rl.mean() * SESIONES_ANIO))
    vol = float(rl.std(ddof=1) * np.sqrt(SESIONES_ANIO))

    acum = np.exp(rl.cumsum())
    max_dd = abs(float((acum / acum.cummax() - 1.0).min()))

    m = dict(
        treynor=((rend_anual - rf) / beta) if beta == beta and beta else np.nan,
        calmar=(rend_anual / max_dd) if max_dd else np.nan,
        m2=np.nan,
        captura_alcista=np.nan, captura_bajista=np.nan, beta_bajista=np.nan,
        hit_ratio=float((r > 0).mean() * 100.0),
        mejor_dia=float(r.max() * 100.0),
        peor_dia=float(r.min() * 100.0),
        asimetria=float(rl.skew()),
        curtosis=float(rl.kurtosis()),
        var_param_95=float((rl.mean() - 1.645 * rl.std(ddof=1)) * 100.0),
    )

    if rend_bench is not None and len(rend_bench) >= 20:
        par = pd.concat([rl, rend_bench], axis=1, join="inner").dropna()
        par.columns = ["p", "b"]
        if len(par) >= 20:
            simples = np.expm1(par)
            sube = simples[simples["b"] > 0]
            baja = simples[simples["b"] < 0]
            if len(sube) and sube["b"].mean():
                m["captura_alcista"] = float(sube["p"].mean() / sube["b"].mean() * 100.0)
            if len(baja) and baja["b"].mean():
                m["captura_bajista"] = float(baja["p"].mean() / baja["b"].mean() * 100.0)
            if len(baja) > 5 and baja["b"].var():
                m["beta_bajista"] = float(baja.cov().iloc[0, 1] / baja["b"].var())
            if vol_bench_anual:
                sharpe = (rend_anual - rf) / vol if vol else np.nan
                if sharpe == sharpe:
                    m["m2"] = float((rf + sharpe * vol_bench_anual) * 100.0)
    return m


def series_moviles(rend_port: pd.Series, rend_bench: pd.Series | None,
                   ventana_vol: int = 30, ventana_beta: int = 60) -> dict:
    """Volatilidad anualizada movil y beta movil contra el benchmark."""
    out = dict(vol=pd.Series(dtype=float), beta=pd.Series(dtype=float))
    if rend_port is None or len(rend_port) < ventana_vol + 5:
        return out
    r = rend_port.dropna()
    out["vol"] = (r.rolling(ventana_vol).std(ddof=1)
                  * np.sqrt(SESIONES_ANIO) * 100.0).dropna()
    if rend_bench is not None and len(rend_bench) >= ventana_beta + 5:
        par = pd.concat([r, rend_bench], axis=1, join="inner").dropna()
        par.columns = ["p", "b"]
        cov = par["p"].rolling(ventana_beta).cov(par["b"])
        var = par["b"].rolling(ventana_beta).var()
        out["beta"] = (cov / var).dropna()
    return out


# --------------------------------------------------------------------------
# Serie real del portafolio (tenencia de cada dia, no la actual)
# --------------------------------------------------------------------------

def serie_valor_real(historico: pd.DataFrame, base: pd.DataFrame,
                     bitacora: pd.DataFrame, efectivo_inicial: float,
                     fecha_base) -> pd.Series:
    """
    Valor diario del portafolio COMPLETO (valores + liquidez) reconstruyendo
    la tenencia vigente en cada fecha desde la posicion base.

    Como las operaciones son internas (la liquidez absorbe cada compra y
    venta), el valor total no tiene flujos externos, y el rendimiento del
    periodo es directamente V_final / V_inicial - 1: un TWR legitimo, a
    diferencia de la serie a titulos constantes que se usa para riesgo.
    """
    if not len(base) or not len(historico):
        return pd.Series(dtype=float)

    fechas = historico.index[historico.index >= pd.Timestamp(fecha_base)]
    if not len(fechas):
        return pd.Series(dtype=float)

    from .taxonomy import ticker_de

    tenencia = {r["ticker"]: float(r["titulos"])
                for _, r in base.iterrows() if r["ticker"]}
    movs = (bitacora.dropna(subset=["fecha"]).sort_values("fecha")
            .to_dict("records") if len(bitacora) else [])

    valores = {}
    efectivo = efectivo_inicial
    i = 0
    for fecha in fechas:
        while i < len(movs) and pd.Timestamp(movs[i]["fecha"]) <= fecha:
            op = movs[i]
            t = ticker_de(op["emisora"])
            if t:
                q = float(op["titulos"])
                delta = q if op["operacion"] == "Compra" else -q
                tenencia[t] = tenencia.get(t, 0.0) + delta
            efectivo += float(op["efecto_caja"])
            i += 1
        fila = historico.loc[fecha]
        total, precio_completo = efectivo, True
        for t, q in tenencia.items():
            if abs(q) < 1e-9:
                continue
            p = fila.get(t)
            if p is None or pd.isna(p):
                precio_completo = False
                continue
            total += q * float(p)
        if precio_completo:
            valores[fecha] = total
    return pd.Series(valores).sort_index()
