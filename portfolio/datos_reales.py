"""
Alimenta el tablero HTML (assets/dashboard_baz.html) con los datos reales.

El HTML trae un componente cuyo estado de ejemplo vive en literales de
JavaScript (SEED, EFECTIVO, métricas de riesgo, atribución, operaciones,
series de las gráficas). Este módulo corre el mismo pipeline que app.py
(posición + boletas + precios vivos + analítica) y sustituye cada literal
por su valor real ANTES de servir la página, mediante reemplazos anclados
que fallan ruidosamente si el HTML cambia de forma.

Los reemplazos se hacen server-side en cada carga (con caché de 5 minutos
en el wrapper), de modo que el HTML sigue siendo autocontenido: el navegador
recibe una página estática cuyo interior ya son cifras reales.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics as an
from . import benchmark as bmk
from . import engine as eng
from . import loader as ld
from . import market as mk
from .taxonomy import BENCHMARK_TICKER, FX_TICKER

RAIZ = Path(__file__).parent.parent

DATOS = RAIZ / "data"
FECHA_BASE = date(2026, 7, 15)

# El tablero usa nombres con acento; la taxonomia interna es ASCII.
SECTOR_UI = {
    "Consumo Basico": "Consumo Básico", "Indice Amplio": "Índice Amplio",
    "Tecnologia": "Tecnología", "Telecomunicaciones": "Telecomunicaciones",
}
REGION_UI = {
    "Mexico": "México", "Japon": "Japón",
    "Desarrollados ex-EE.UU.": "Des. ex-EE.UU.",
}
CLASE_UI = {"Accion": "Acción"}
MES_UI = ["", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _ui(mapa: dict, v: str) -> str:
    return mapa.get(v, v)


# --------------------------------------------------------------------------
# Serializador a literales de JavaScript
# --------------------------------------------------------------------------
# El componente vive dentro de una cadena JS del bundle donde las comillas
# dobles van escapadas; serializar con comillas simples y ASCII puro evita
# tocar ese esquema de escape.

def js(v) -> str:
    if isinstance(v, str):
        # Contexto de doble escape: el componente es a su vez una cadena JS
        # del bundle, que decodifica una vez. Un apostrofe debe llegar alli
        # como \' , por lo que aqui se emite \\' ; los no-ASCII van como
        # \uXXXX, que el nivel exterior decodifica al caracter real (valido
        # dentro de la cadena interior). Nunca se emiten comillas dobles.
        cuerpo = "".join(
            c if 32 <= ord(c) < 127 and c not in "'\\\"" else
            ("\\\\'" if c == "'" else "\\\\\\\\" if c == "\\" else
             f"\\u{ord(c):04x}")
            for c in v)
        return f"'{cuerpo}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "0"
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(js(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(
            (k if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) else js(k))
            + ":" + js(x) for k, x in v.items()) + "}"
    raise TypeError(f"No serializable a JS: {type(v)}")


def _fM(v: float) -> str:
    """Replica el formateador fM del componente: '$-3.06 M'."""
    return f"${v/1e6:,.2f} M"


# --------------------------------------------------------------------------
# Pipeline: identico al de app.py
# --------------------------------------------------------------------------

def calcular(ventana: int = 252, tasa_libre: float = 0.09,
             efectivo_inicial: float = 0.0) -> dict:
    pos_path = next(p for p in sorted(DATOS.glob("*.xlsx"))
                    if "posici" in p.name.lower())
    base = ld.leer_posicion_base(pos_path.read_bytes(), "Hoja1")

    bloques = [ld.leer_operaciones(p.read_bytes(), fuente=f"boleta {p.stem[:14]}")
               for p in sorted(DATOS.glob("*.xlsx")) if p != pos_path]
    bloques.append(ld.leer_movimientos(pos_path.read_bytes(), None))
    movimientos = ld.consolidar_movimientos(*bloques)

    res = eng.construir_posicion(base.df, movimientos, efectivo_inicial)

    bench_df = bmk.cargar_benchmark()
    tickers = tuple(t for t in res.posiciones["ticker"] if t)
    universo = tuple(dict.fromkeys(
        tickers + bmk.tickers_benchmark(bench_df)
        + (BENCHMARK_TICKER, FX_TICKER)))

    vigentes = mk.precios_vigentes(tickers)
    historico = mk.descargar_historico(universo, dias=int(ventana * 1.5))
    precios, previos = mk.mapas_de_precio(vigentes)

    val = eng.valuar(res.posiciones, precios, previos)
    resumen = eng.resumen_cartera(val, res.efectivo, res.realizado)

    ha = historico.drop(columns=[BENCHMARK_TICKER, FX_TICKER], errors="ignore")
    bench_hist = historico[BENCHMARK_TICKER].dropna()
    fx_hist = historico[FX_TICKER].dropna()

    rend_port = an.rendimientos_portafolio(ha, val)
    rend_bench = np.log(bench_hist / bench_hist.shift(1)).dropna()
    rend_fx = np.log(fx_hist / fx_hist.shift(1)).dropna()

    riesgo = an.metricas_riesgo(rend_port, rend_bench, tasa_libre)
    vol_bench = float(rend_bench.std(ddof=1) * np.sqrt(252))
    ext = an.metricas_extendidas(rend_port, rend_bench, tasa_libre,
                                 riesgo["beta"], vol_bench)
    conc = an.metricas_concentracion(val)
    rpos = an.descomposicion_riesgo(ha, val)
    fx = an.exposicion_fx(val)

    sv_real = an.serie_valor_real(historico, base.df, res.bitacora,
                                  efectivo_inicial, FECHA_BASE)
    twr = (float(sv_real.iloc[-1] / sv_real.iloc[0] - 1.0) * 100.0
           if len(sv_real) >= 2 and sv_real.iloc[0] else None)
    b_per = bench_hist[bench_hist.index >= pd.Timestamp(FECHA_BASE)]
    ipc_per = (float(b_per.iloc[-1] / b_per.iloc[0] - 1.0) * 100.0
               if len(b_per) >= 2 else None)

    return dict(base=base, res=res, val=val, resumen=resumen,
                historico=historico, ha=ha, bench_hist=bench_hist,
                rend_port=rend_port, rend_bench=rend_bench, rend_fx=rend_fx,
                riesgo=riesgo, ext=ext, conc=conc, rpos=rpos, fx=fx,
                bench_df=bench_df, twr=twr, ipc_per=ipc_per,
                tasa_libre=tasa_libre)


# --------------------------------------------------------------------------
# Constructores de cada bloque inyectado
# --------------------------------------------------------------------------

def _seed(val: pd.DataFrame) -> list:
    filas = []
    for _, r in val.iterrows():
        filas.append([
            r["emisora"], _ui(SECTOR_UI, r["sector"]),
            _ui(REGION_UI, r["region"]), r["mercado"],
            _ui(CLASE_UI, r["clase_activo"]),
            float(r["precio_mercado"]),
            float(r["valor_mercado"]) / 1e6,
            float(r["rend_pct"]),
            float(r["var_dia_pct"]) if r["var_dia_pct"] == r["var_dia_pct"] else 0.0,
            r["divisa_subyacente"],
        ])
    return filas


def _grafica_desempeno(ha, val, bench_hist, n_max: int = 180) -> dict:
    """Puntos SVG reales para la linea portafolio vs IPC (base 100)."""
    serie = an.serie_valor_portafolio(ha, val).tail(n_max)
    ipc = bench_hist.reindex(serie.index).ffill().dropna()
    serie = serie.reindex(ipc.index)
    if len(serie) < 10:
        raise ValueError("Serie de desempeno insuficiente")

    p = (serie / serie.iloc[0] * 100.0).values
    b = (ipc / ipc.iloc[0] * 100.0).values
    n = len(p)
    lo = float(np.floor(min(p.min(), b.min()) - 1))
    hi = float(np.ceil(max(p.max(), b.max()) + 1))

    W, H, padL, padR, padT, padB = 1000, 300, 46, 66, 14, 30
    X = lambda i: padL + i / (n - 1) * (W - padL - padR)
    Y = lambda v: padT + (hi - v) / (hi - lo) * (H - padT - padB)
    pts = lambda a: " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(a))

    marcas = np.linspace(lo, hi, 5)
    y_ticks = [dict(label=f"{v:.0f}", y=f"{Y(v):.1f}") for v in marcas]
    idx = serie.index
    posiciones = [0, int((n - 1) * 0.34), int((n - 1) * 0.66), n - 1]
    x_ticks = [dict(label=f"{MES_UI[idx[i].month]} {idx[i].year}",
                    x=f"{X(i):.1f}") for i in posiciones]

    chart = dict(portPoints=pts(p), ipcPoints=pts(b),
                 yTicks=y_ticks, xTicks=x_ticks,
                 endX=f"{X(n-1):.1f}", portEndY=f"{Y(p[-1]):.1f}",
                 ipcEndY=f"{Y(b[-1]):.1f}")

    # Serie para el crosshair del runtime de tooltips: un texto por sesion.
    tips = [f"{f.day} {MES_UI[f.month]} {f.year}|Portafolio {v:.2f}|IPC {w:.2f}"
            for f, v, w in zip(idx, p, b)]
    serie_hover = dict(W=1000, padL=46, padR=66, n=n, tips=tips)
    return chart, serie_hover


def _bloque_riesgo(d: dict) -> dict:
    """hist / stress / factors / drawdown con datos reales."""
    r = np.expm1(d["rend_port"].dropna())
    riesgo, fx, resumen = d["riesgo"], d["fx"], d["resumen"]
    valor_instr = resumen["valor_instrumentos"]

    # --- histograma de rendimientos diarios (23 casillas) ---------------
    bins = 23
    lim = float(max(abs(r.min()), abs(r.max())) * 1.05)
    bordes = np.linspace(-lim, lim, bins + 1)
    conteo, _ = np.histogram(r.values, bins=bordes)
    mx = conteo.max() or 1
    var95 = riesgo["var_95"] / 100.0
    var99 = riesgo["var_99"] / 100.0
    hist = []
    for i, c in enumerate(conteo):
        x = (bordes[i] + bordes[i + 1]) / 2
        color = ("var(--neg)" if x <= var99 else
                 "color-mix(in srgb, var(--neg) 55%, transparent)" if x <= var95
                 else "var(--brand-med)" if x < 0 else "var(--brand-lite)")
        hist.append(dict(
            h=f"{c / mx * 100:.0f}", color=color,
            tip=(f"{bordes[i]*100:+.2f} % a {bordes[i+1]*100:+.2f} %"
                 f"|{int(c)} sesiones")))

    # --- escenarios parametricos con las sensibilidades reales -----------
    beta = riesgo["beta"] if riesgo["beta"] == riesgo["beta"] else 1.0
    usd_frac = fx["usd_pct"] / 100.0
    escenarios = [
        ("IPC −10 %", beta * -10.0),
        ("IPC −20 %", beta * -20.0),
        ("USDMXN +10 %", usd_frac * 10.0),
        ("USDMXN −10 %", usd_frac * -10.0),
        ("Mar-2020: IPC −27 %, FX +25 %", beta * -27.0 + usd_frac * 25.0),
        ("Jun-2024: IPC −6 %, FX +8 %", beta * -6.0 + usd_frac * 8.0),
    ]
    smax = max(abs(v) for _, v in escenarios)
    stress = [dict(name=n_, pct=f"{v:+.1f} %", monto=_fM(valor_instr * v / 100),
                   w=f"{abs(v) / smax * 100:.0f}",
                   color="var(--pos)" if v >= 0 else "var(--neg)",
                   tip=(f"{n_}|Impacto {v:+.1f} % del patrimonio"
                        f"|{_fM(valor_instr * v / 100)}"))
              for n_, v in escenarios]

    # --- cargas factoriales por regresion (IPC y USDMXN) ----------------
    par = pd.concat([d["rend_port"], d["rend_bench"], d["rend_fx"]],
                    axis=1, join="inner").dropna()
    par.columns = ["p", "b", "f"]
    b_ipc = b_fx = np.nan
    if len(par) > 40:
        Xm = np.column_stack([np.ones(len(par)), par["b"], par["f"]])
        coef, *_ = np.linalg.lstsq(Xm, par["p"].values, rcond=None)
        b_ipc, b_fx = float(coef[1]), float(coef[2])
    factores = [
        ("Mercado (β IPC)", b_ipc),
        ("USD/MXN (β FX)", b_fx),
        ("β bajista", d["ext"]["beta_bajista"]),
        ("Correlación IPC", riesgo["correlacion"]),
    ]
    factores = [(n_, v) for n_, v in factores if v == v]
    fmax = max(abs(v) for _, v in factores) or 1.0
    factors = [dict(name=n_, valF=f"{v:+.2f}",
                    wpos=f"{v / fmax * 100:.0f}" if v > 0 else 0,
                    wneg=f"{abs(v) / fmax * 100:.0f}" if v < 0 else 0,
                    color="var(--brand-lite)" if v >= 0 else "var(--warn)",
                    tip=f"{n_}|Carga {v:+.2f}")
               for n_, v in factores]

    # --- drawdown real ----------------------------------------------------
    serie = an.serie_valor_portafolio(d["ha"], d["val"]).tail(180)
    dd = (serie / serie.cummax() - 1.0).values * 100.0
    n = len(dd)
    min_dd = float(dd.min())
    lo2 = min(-10.0, min_dd * 1.12)
    W, H, padT, padB = 1000, 170, 8, 6
    Xd = lambda i: f"{i / (n - 1) * W:.1f}"
    Yd = lambda v: f"{padT + (0 - v) / (0 - lo2) * (H - padT - padB):.1f}"
    linea = " ".join(f"{Xd(i)},{Yd(v)}" for i, v in enumerate(dd))
    path = f"M0,{Yd(0)} {linea} L{W},{Yd(0)} Z"

    bloque = dict(hist=hist, stress=stress, factors=factors,
                  ddLine=linea, ddPath=path, ddMinF=f"{min_dd:.1f} %",
                  ddZeroY=Yd(0))
    idx = serie.index
    tips_dd = [f"{f.day} {MES_UI[f.month]} {f.year}|Caída {v:.2f} %"
               for f, v in zip(idx, dd)]
    serie_dd = dict(W=1000, padL=0, padR=0, n=n, tips=tips_dd)

    # --- capa explicativa: marcadores, eje y notas de lectura -------------
    # El histograma del VaR es la grafica de riesgo central y venia sin eje
    # ni marcadores; cada nota traduce la metrica a una lectura de mesa con
    # los montos del mandato.
    def _posx(x: float) -> float:
        return max(1.5, min(98.5, (x + lim) / (2 * lim) * 100.0))

    monto95 = valor_instr * abs(var95)
    monto99 = valor_instr * abs(var99)
    cvar = riesgo["cvar_95"] / 100.0
    n_ses = int(len(r))

    beta_txt = f"{beta:.2f}"
    usd_txt = f"{fx['usd_pct']:.0f} %"

    corr8 = an.matriz_correlacion(d["ha"], d["val"], top=8)
    prom_corr = float("nan")
    par_min = ""
    if len(corr8) >= 2:
        m_ = corr8.values
        fuera_diag = m_[~np.eye(len(m_), dtype=bool)]
        prom_corr = float(fuera_diag.mean())
        i_, j_ = np.unravel_index(np.argmin(m_ + np.eye(len(m_)) * 2), m_.shape)
        par_min = (f"{corr8.index[i_]} y {corr8.columns[j_]} "
                   f"({m_[i_, j_]:+.2f})")

    top_riesgo = d["rpos"].iloc[0] if len(d["rpos"]) else None
    fecha_dd = idx[int(np.argmin(dd))]

    bloque.update(
        var95X=round(_posx(var95), 2), var99X=round(_posx(var99), 2),
        var95F=f"{riesgo['var_95']:.2f} %", var99F=f"{riesgo['var_99']:.2f} %",
        cvarF=f"{riesgo['cvar_95']:.2f} %",
        var95Tip=(f"VaR 95 % (histórico)|Pérdida diaria que sólo se excede "
                  f"1 de cada 20 sesiones|{riesgo['var_95']:.2f} % "
                  f"≈ {_fM(monto95)}"),
        var99Tip=(f"VaR 99 % (histórico)|Pérdida diaria que sólo se excede "
                  f"1 de cada 100 sesiones|{riesgo['var_99']:.2f} % "
                  f"≈ {_fM(monto99)}"),
        ejeLo=f"{-lim*100:.1f}%", ejeMedLo=f"{-lim*50:.1f}%",
        ejeMedHi=f"+{lim*50:.1f}%", ejeHi=f"+{lim*100:.1f}%",
        notaVar=(f"Distribución de los últimos {n_ses} rendimientos diarios. "
                 f"Lectura: con 95 % de confianza la pérdida de un día no "
                 f"excede {riesgo['var_95']:.2f} % (≈ {_fM(monto95)} del "
                 f"mandato); en la peor sesión de cada 100 supera "
                 f"{riesgo['var_99']:.2f} % (≈ {_fM(monto99)}). Cuando la "
                 f"pérdida cae en la cola del 5 %, el promedio (CVaR) es "
                 f"{riesgo['cvar_95']:.2f} % ≈ {_fM(valor_instr * abs(cvar))}."),
        notaStress=(f"Impacto estimado con las sensibilidades observadas del "
                    f"portafolio: β {beta_txt} contra el IPC y {usd_txt} de "
                    f"exposición USD. Pérdida esperada = β × choque de índice "
                    f"+ exposición USD × choque cambiario."),
        notaFactores=(f"Sensibilidades por regresión sobre la ventana: un "
                      f"movimiento de −1 % del IPC mueve al portafolio "
                      f"≈ {b_ipc:+.2f} % y una depreciación de 1 % del peso "
                      f"lo mueve ≈ {b_fx:+.2f} %."
                      if b_ipc == b_ipc else
                      "Sensibilidades por regresión sobre la ventana."),
        notaPesoRiesgo=(f"Contribución marginal al riesgo con la matriz de "
                        f"covarianzas: una barra de riesgo mayor que la de "
                        f"peso señala sobreconsumo del presupuesto. Hoy la "
                        f"mayor es {top_riesgo['emisora']} con "
                        f"{top_riesgo['contrib_riesgo_pct']:.1f} % del riesgo "
                        f"sobre {top_riesgo['peso_pct']:.1f} % del peso."
                        if top_riesgo is not None else ""),
        notaCorr=(f"Correlación de rendimientos diarios entre las ocho "
                  f"mayores posiciones (promedio {prom_corr:+.2f}). Valores "
                  f"altos restan diversificación; el par más defensivo es "
                  f"{par_min}."
                  if prom_corr == prom_corr else ""),
        notaScatter=("Cada burbuja es una posición; el tamaño es su peso. "
                     "El cuadrante deseable es arriba a la izquierda: más "
                     "rendimiento por unidad de volatilidad."),
        notaDd=(f"Caída acumulada desde el máximo previo de la cartera "
                f"vigente. El peor punto de la ventana fue {min_dd:.1f} % "
                f"el {fecha_dd.day} {MES_UI[fecha_dd.month]} {fecha_dd.year}."),
    )
    return bloque, serie_dd


def _atribucion_periodos(d: dict) -> dict:
    """Brinson-Fachler real para los cuatro periodos del selector."""
    out = {}
    for etiqueta, ses in [("1 mes", 21), ("3 meses", 63),
                          ("6 meses", 126), ("12 meses", 252)]:
        bs = bmk.sectores_benchmark(d["bench_df"], d["historico"], ses)
        bf = an.brinson_fachler(d["val"], d["historico"], bs, ses)
        tabla = [[_ui(SECTOR_UI, t.sector),
                  round(t.w_p_pct, 1), round(t.w_b_pct, 1),
                  round(t.asignacion_pp, 2), round(t.seleccion_pp, 2),
                  round(t.interaccion_pp, 2), round(t.total_pp, 2)]
                 for t in bf["tabla"].itertuples()]
        out[etiqueta] = dict(
            r_b=round(bf["r_b"], 2), r_p=round(bf["r_p"], 2),
            asig=round(bf["asignacion"], 2), sel=round(bf["seleccion"], 2),
            inter=round(bf["interaccion"], 2), activo=round(bf["activo"], 2),
            tabla=tabla)
    return out


def _correlaciones(d: dict) -> tuple[list, list]:
    corr = an.matriz_correlacion(d["ha"], d["val"], top=8)
    if not len(corr):
        return [], []
    nombres = [str(c) for c in corr.columns]
    etiquetas = [nom.split(" ")[0][:5] for nom in nombres]
    filas = []
    for i, lab in enumerate(etiquetas):
        celdas = []
        for j in range(len(etiquetas)):
            v = float(corr.iloc[i, j])
            celdas.append(dict(
                v=f"{v:.2f}",
                bg=f"rgba(144,133,233,{0.1 + max(v, 0) * 0.7:.2f})",
                txt="#fff" if v > 0.6 else "var(--ink2)",
                tip=f"{nombres[i]} vs {nombres[j]}|Correlación {v:+.2f}"))
        filas.append(dict(label=lab, cells=celdas))
    return etiquetas, filas


def _operaciones(res) -> list:
    ops = []
    for _, o in res.bitacora.iterrows():
        f = o["fecha"]
        etiqueta = f"{f.day} {MES_UI[f.month]}" if f is not None and f == f else "—"
        ops.append([etiqueta, "C" if o["operacion"] == "Compra" else "V",
                    o["emisora"], float(o["titulos"]), float(o["precio"])])
    return ops


def _hallazgos(d: dict) -> list:
    estilo = {"critico": ("‼", "var(--neg)"),
              "atencion": ("▲", "var(--warn)"),
              "favorable": ("✓", "var(--pos)")}
    tecnicos = mk.indicadores_tecnicos(d["ha"])
    lista = an.diagnostico(d["val"], d["riesgo"], d["conc"], tecnicos,
                           d["rpos"], d["resumen"]["peso_efectivo_pct"])
    out = []
    for h in lista[:6]:
        icono, color = estilo.get(h["severidad"], ("•", "var(--ink2)"))
        out.append(dict(icon=icono, color=color,
                        titulo=h["titulo"], detalle=h["detalle"]))
    return out


# --------------------------------------------------------------------------
# Inyeccion
# --------------------------------------------------------------------------

def _sub(html: str, ancla: str, nuevo: str, veces: int = 1) -> str:
    n = html.count(ancla)
    assert n == veces, f"Ancla {ancla[:60]!r} aparece {n} veces (se esperaba {veces})"
    return html.replace(ancla, nuevo)


def _sub_re(html: str, patron: str, nuevo: str) -> str:
    rx = re.compile(patron, re.S)
    encontrados = rx.findall(html)
    assert len(encontrados) == 1, \
        f"Patron {patron[:60]!r}: {len(encontrados)} coincidencias"
    return rx.sub(lambda _m: nuevo, html, count=1)


def inyectar(html: str, d: dict) -> str:
    val, res, resumen = d["val"], d["res"], d["resumen"]
    riesgo, ext, conc, fx = d["riesgo"], d["ext"], d["conc"], d["fx"]

    # --- 1. datos semilla -------------------------------------------------
    html = _sub_re(html, r"SEED = \[.*?\];", f"SEED = {js(_seed(val))};")
    html = _sub(html, "EFECTIVO = 23_640_000;",
                f"EFECTIVO = {res.efectivo!r};")

    # --- 2. KPIs de cabecera ---------------------------------------------
    twr_txt = (f"TWR {d['twr']:+.2f} %" if d["twr"] is not None else "TWR —")
    ipc_txt = (f"IPC {d['ipc_per']:+.2f} %" if d["ipc_per"] is not None else "")
    html = _sub(html, "delta: 'vs IPC +7.10 %'",
                f"delta: {js(twr_txt + ' · ' + ipc_txt)}")

    alpha = riesgo["alpha_anual"]
    col_a = "var(--pos)" if alpha >= 0 else "var(--neg)"
    html = _sub(
        html,
        "{ label: 'Alpha anual', value: '+3.4 %', delta: 'Jensen · vs IPC', color: 'var(--pos)' },",
        f"{{ label: 'Alpha anual', value: {js(f'{alpha:+.1f} %')}, "
        f"delta: 'Jensen · vs IPC', color: '{col_a}' }},")

    var95_txt = js(f"{riesgo['var_95']:+.2f} %")
    var95_frac = abs(riesgo["var_95"]) / 100.0
    html = _sub(
        html,
        "{ label: 'VaR 95% diario', value: '-1.62 %', delta: this.fM(valorInstr * 0.0162), color: 'var(--neg)' },",
        f"{{ label: 'VaR 95% diario', value: {var95_txt}, "
        f"delta: this.fM(valorInstr * {var95_frac:.6f}), color: 'var(--neg)' }},")

    sharpe_txt = js(f"{riesgo['sharpe']:.2f}")
    beta_vol_txt = js(f"β {riesgo['beta']:.2f} · vol {riesgo['vol_anual']:.1f} %")
    html = _sub(
        html,
        "{ label: 'Sharpe / Beta', value: '1.34', delta: 'β 0.92 · vol 12.8 %', color: 'var(--ink2)' },",
        f"{{ label: 'Sharpe / Beta', value: {sharpe_txt}, "
        f"delta: {beta_vol_txt}, color: 'var(--ink2)' }},")

    # --- 3. bloques de riesgo --------------------------------------------
    def _t(label, value, note, color):
        return dict(label=label, value=value, note=note, color=color)

    def _f2(v, suf="", signo=True, dec=2):
        if v != v:
            return "—"
        return f"{v:+.{dec}f}{suf}" if signo else f"{v:.{dec}f}{suf}"

    resumen_risk = [
        _t("Volatilidad", _f2(riesgo["vol_anual"], " %", False, 1), "anual", "var(--ink)"),
        _t("Sharpe", _f2(riesgo["sharpe"], "", False), f"Cetes {d['tasa_libre']:.1%}", "var(--ink)"),
        _t("Beta vs IPC", _f2(riesgo["beta"], "", False), "", "var(--ink)"),
        _t("Caída máxima", _f2(riesgo["max_drawdown"], " %", True, 1), "", "var(--neg)"),
        _t("Pos. efectivas", _f2(conc["n_efectivo"], "", False, 1),
           f"top 5: {conc['top5_pct']:.0f}%", "var(--ink)"),
        _t("Herfindahl", f"{conc['hhi']:,.0f}", "0–10,000", "var(--ink)"),
    ]
    html = _sub_re(html, r"const resumenRisk = \[.*?\];",
                   f"const resumenRisk = {js(resumen_risk)};")

    risk_tiles = [
        _t("Volatilidad", _f2(riesgo["vol_anual"], " %", False, 1), "anual 252d", "var(--ink)"),
        _t("Sharpe", _f2(riesgo["sharpe"], "", False), "", "var(--ink)"),
        _t("Sortino", _f2(riesgo["sortino"], "", False), "", "var(--ink)"),
        _t("Beta", _f2(riesgo["beta"], "", False), "vs IPC", "var(--ink)"),
        _t("Alfa anual", _f2(alpha, " %", True, 1), "",
           "var(--pos)" if alpha >= 0 else "var(--neg)"),
        _t("Tracking error", _f2(riesgo["tracking_error"], " %", False, 1), "", "var(--ink)"),
        _t("Info ratio", _f2(riesgo["info_ratio"], "", False), "", "var(--ink)"),
        _t("VaR 95%", _f2(riesgo["var_95"], " %"), "hist. diario", "var(--neg)"),
        _t("VaR 99%", _f2(riesgo["var_99"], " %"), "", "var(--neg)"),
        _t("CVaR 95%", _f2(riesgo["cvar_95"], " %"), "cola media", "var(--neg)"),
        _t("Caída máxima", _f2(riesgo["max_drawdown"], " %", True, 1), "", "var(--neg)"),
        _t("Treynor", _f2(ext["treynor"], "", False), "exc./beta", "var(--ink)"),
        _t("Calmar", _f2(ext["calmar"], "", False), "rend/DD", "var(--ink)"),
        _t("M²", _f2(ext["m2"], " %", True, 1), "a vol IPC",
           "var(--pos)" if ext["m2"] == ext["m2"] and ext["m2"] >= 0 else "var(--neg)"),
        _t("Captura ↑", _f2(ext["captura_alcista"], " %", False, 0), "días + IPC", "var(--ink)"),
        _t("Captura ↓", _f2(ext["captura_bajista"], " %", False, 0), "menor mejor", "var(--ink)"),
        _t("Hit ratio", _f2(ext["hit_ratio"], " %", False, 0), "días +", "var(--ink)"),
        _t("Asimetría", _f2(ext["asimetria"]), "", "var(--ink)"),
        _t("Curtosis", _f2(ext["curtosis"]), "colas", "var(--ink)"),
        _t("Top 10", _f2(conc["top10_pct"], " %", False, 1), "concentración", "var(--ink)"),
    ]
    html = _sub_re(html, r"const riskTiles = \[.*?\];",
                   f"const riskTiles = {js(risk_tiles)};")

    # peso vs contribucion al riesgo (contribucion marginal real)
    rpos = d["rpos"].head(8)
    max_rw = float(max(rpos["peso_pct"].max(),
                       rpos["contrib_riesgo_pct"].max())) * 1.15
    risk_weight = [dict(name=str(r.emisora), pesoF=f"{r.peso_pct:.1f}%",
                        riskF=f"{r.contrib_riesgo_pct:.1f}%",
                        wPeso=round(r.peso_pct / max_rw * 100, 1),
                        wRisk=round(r.contrib_riesgo_pct / max_rw * 100, 1),
                        tip=(f"{r.emisora}|Peso {r.peso_pct:.2f} %"
                             f"|Contribución al riesgo {r.contrib_riesgo_pct:.2f} %"
                             f"|Riesgo/peso {r.riesgo_sobre_peso:.2f}x"))
                   for r in rpos.itertuples()]
    html = _sub_re(html, r"const riskTop = rows\.slice\(0, 8\);.*?\}\);",
                   f"const riskWeight = {js(risk_weight)};")

    # dispersion vol individual vs rendimiento (real)
    disp = d["val"].merge(d["rpos"][["ticker", "vol_individual_pct"]],
                          on="ticker", how="inner")
    vmin, vmax = disp["vol_individual_pct"].min(), disp["vol_individual_pct"].max()
    rmin, rmax = disp["rend_pct"].min(), disp["rend_pct"].max()
    dv = (vmax - vmin) or 1.0
    dr = (rmax - rmin) or 1.0
    scatter = [dict(cx=f"{30 + (r.vol_individual_pct - vmin) / dv * 360:.0f}",
                    cy=f"{185 - (r.rend_pct - rmin) / dr * 170:.0f}",
                    r=f"{3 + np.sqrt(r.peso_pct) * 3.4:.1f}",
                    color="var(--brand-lite)" if r.rend_pct >= 0 else "var(--neg)",
                    tip=(f"{r.emisora}|Volatilidad {r.vol_individual_pct:.1f} %"
                         f"|Rendimiento {r.rend_pct:+.2f} %"
                         f"|Peso {r.peso_pct:.2f} %"))
               for r in disp.itertuples()]
    html = _sub_re(html, r"const scatter = rows\.map\(r => \{.*?\}\);",
                   f"const scatter = {js(scatter)};")

    # correlaciones reales
    etiquetas, corr_rows = _correlaciones(d)
    html = _sub_re(html, r"const corrLabelsFull = .*?\}\)\);",
                   f"const corrLabelsFull = {js(etiquetas)}; "
                   f"const corrRows = {js(corr_rows)};")

    # --- 4. atribucion por periodo (el selector se vuelve funcional) -----
    periodos = _atribucion_periodos(d)
    html = _sub(
        html,
        "const r_b = 3.11, asig = 0.18, sel = 1.28, inter = 0.25, r_p = 4.82, activo = 1.71;",
        f"const _A = ({js(periodos)})[this.state.period]; "
        "const r_b=_A.r_b, asig=_A.asig, sel=_A.sel, inter=_A.inter, "
        "r_p=_A.r_p, activo=_A.activo;")
    html = _sub_re(html, r"const attrData = \[.*?\];",
                   "const attrData = _A.tabla;")
    html = _sub(html, "const scale = 170 / 6;",
                "const scale = 170 / Math.max(6, Math.abs(r_p) * 1.25, "
                "Math.abs(r_b) * 1.25, Math.abs(r_b + asig) * 1.25, "
                "Math.abs(r_b + asig + sel) * 1.25);")
    html = _sub(html, "value: '+' + activo.toFixed(2) + ' pp', note: '', color: 'var(--pos)' }",
                "value: (activo >= 0 ? '+' : '') + activo.toFixed(2) + ' pp', note: '', color: this.col(activo) }")
    html = _sub(html, "value: '+' + asig.toFixed(2) + ' pp'",
                "value: (asig >= 0 ? '+' : '') + asig.toFixed(2) + ' pp'")
    html = _sub(html, "value: '+' + sel.toFixed(2) + ' pp'",
                "value: (sel >= 0 ? '+' : '') + sel.toFixed(2) + ' pp'")
    html = _sub(html, "value: '+' + inter.toFixed(2) + ' pp'",
                "value: (inter >= 0 ? '+' : '') + inter.toFixed(2) + ' pp'")


    # --- 5. liquidez y operaciones ---------------------------------------
    html = _sub(html, "this.fM(31_196_000)", f"this.fM({res.flujo_ventas!r})")
    html = _sub(html, "this.fM(-46_734_000)", f"this.fM({-res.flujo_compras!r})")
    html = _sub(html, "this.fMoney(-31_172)", f"this.fMoney({-res.costos_totales!r})")
    html = _sub(html, "this.fM(1_284_000)", f"this.fM({res.realizado!r})")
    html = _sub_re(html, r"const OPS = \[.*?\];",
                   f"const OPS = {js(_operaciones(res))};")

    # --- 6. graficas precalculadas ---------------------------------------
    chart, serie_perf = _grafica_desempeno(d["ha"], val, d["bench_hist"])
    html = _sub(html, "const chart = this.buildChart();",
                f"const chart = {js(chart)};")
    bloque_riesgo, serie_dd = _bloque_riesgo(d)
    html = _sub(html, "const risk = this.buildRisk(valorInstr);",
                f"const risk = {js(bloque_riesgo)};")

    # --- 7. diagnostico ---------------------------------------------------
    html = _sub_re(html, r"const hallazgos = \[.*?\];",
                   f"const hallazgos = {js(_hallazgos(d))};")

    # --- 7b. riesgo explicativo (antes de la capa interactiva) ------------
    html = _riesgo_explicativo(html)

    # --- 8. capa interactiva: tooltips y crosshair ------------------------
    html = _capa_interactiva(html, d, serie_perf, serie_dd)
    html = _tabla_posiciones(html)

    # --- 9. sello de fecha/hora en la cabecera ---------------------------
    ahora = datetime.now()
    html = _sub(html, "21 jul 2026",
                f"{ahora.day} {MES_UI[ahora.month].lower()} {ahora.year}")
    html = re.sub(r"14:32:\d{2}", ahora.strftime("%H:%M:%S"), html, count=1)

    return html


# --------------------------------------------------------------------------
# Capa interactiva: tooltips en cada elemento y crosshair en las lineas
# --------------------------------------------------------------------------
# El tablero es HTML/SVG estatico y no traia informacion al pasar el cursor.
# Esta capa agrega (1) atributos data-tip en el template, ligados a campos
# `tip` que se anaden a cada dato, y (2) un runtime delegado que muestra un
# tooltip siguiendo al cursor y un crosshair con fecha y valores sobre las
# graficas de linea. El runtime va en el documento exterior, fuera del
# bundle, por lo que sobrevive a los re-renders del componente.

def _capa_interactiva(html: str, d: dict, serie_perf: dict,
                      serie_dd: dict) -> str:
    # ---- 1. atributos data-tip en el template (marcado escapado) ---------
    plantilla = [
        # treemap: mosaico por emisora y cabecera de sector
        ('<div style=\\"{{ t.stl }}\\">',
         '<div data-tip=\\"{{ t.tip }}\\" style=\\"{{ t.stl }}\\">', 1),
        ('<div style=\\"{{ sec.headerStl }}\\">',
         '<div data-tip=\\"{{ sec.tip }}\\" style=\\"{{ sec.headerStl }}\\">', 1),
        # barras de contribucion (rendimiento y dia comparten marcado)
        ('<div style=\\"display:flex;align-items:center;gap:8px;margin-bottom:4px\\">',
         '<div data-tip=\\"{{ c.tip }}\\" style=\\"display:flex;align-items:center;gap:8px;margin-bottom:4px\\">', 2),
        # asignacion por dimension
        ('<div style=\\"display:flex;align-items:center;gap:10px;margin-bottom:11px\\">',
         '<div data-tip=\\"{{ b.tip }}\\" style=\\"display:flex;align-items:center;gap:10px;margin-bottom:11px\\">', 1),
        # dispersion riesgo-rendimiento
        ('<circle cx=\\"{{ p.cx }}\\"',
         '<circle data-tip=\\"{{ p.tip }}\\" cx=\\"{{ p.cx }}\\"', 1),
        # celdas de correlacion
        (';margin:1px\\">', ';margin:1px\\" data-tip=\\"{{ cell.tip }}\\">', 1),
        # escenarios de estres
        ('gap:10px;margin-bottom:9px\\">',
         'gap:10px;margin-bottom:9px\\" data-tip=\\"{{ s.tip }}\\">', 1),
        # cargas factoriales
        ('gap:8px;margin-bottom:7px\\">',
         'gap:8px;margin-bottom:7px\\" data-tip=\\"{{ f.tip }}\\">', 1),
        # histograma de rendimientos
        ('border-radius:2px 2px 0 0;min-height:2px\\">',
         'border-radius:2px 2px 0 0;min-height:2px\\" data-tip=\\"{{ b.tip }}\\">', 1),
        # peso vs contribucion al riesgo
        ('gap:8px;margin-bottom:8px\\">',
         'gap:8px;margin-bottom:8px\\" data-tip=\\"{{ r.tip }}\\">', 1),
        # cascada de atribucion
        ('flex:1;height:100%;position:relative\\">',
         'flex:1;height:100%;position:relative\\" data-tip=\\"{{ w.tip }}\\">', 1),
        # flujo diario de liquidez
        ('justify-content:center;position:relative\\">',
         'justify-content:center;position:relative\\" data-tip=\\"{{ f.tip }}\\">', 1),
        # lineas con crosshair
        ('points=\\"{{ chart.portPoints }}\\"',
         'data-chart=\\"perf\\" points=\\"{{ chart.portPoints }}\\"', 1),
        ('points=\\"{{ risk.ddLine }}\\"',
         'data-chart=\\"dd\\" points=\\"{{ risk.ddLine }}\\"', 1),
    ]
    for ancla, nuevo, veces in plantilla:
        html = _sub(html, ancla, nuevo, veces)

    # ---- 2. campos tip en los datos que construye el propio componente ---
    html = _sub(html,
        "const all = rows.map(r => ({ name: r.emisora, val: getter(r) / base * 100 }));",
        "const all = rows.map(r => ({ name: r.emisora, val: getter(r) / base * 100, raw: getter(r) }));")
    html = _sub(html,
        "const bars = top.map(c => ({ name: c.name, valF: this.fPct(c.val), color: this.col(c.val),",
        "const bars = top.map(c => ({ name: c.name, valF: this.fPct(c.val), color: this.col(c.val), "
        "tip: c.name + '|Contribuci\\u00f3n ' + this.fPct(c.val) + ' pp|' + this.fMoney(c.raw),")
    html = _sub(html,
        "const segBars = segG.map((g, i) => ({ name: g.name, pesoF: g.peso.toFixed(1) + '%',",
        "const segBars = segG.map((g, i) => ({ name: g.name, pesoF: g.peso.toFixed(1) + '%', "
        "tip: g.name + '|' + this.fM(g.valor) + ' \\u00b7 ' + g.peso.toFixed(2) + ' %|P&L ' + this.fM(g.pnl) + ' (' + this.fPct(g.rend) + ')|' + g.n + ' posiciones',")
    html = _sub(html,
        ".map(r => ({ value: r.valor, emisora: r.emisora, peso: r.peso }))",
        ".map(r => ({ value: r.valor, emisora: r.emisora, peso: r.peso, "
        "tip: r.emisora + '|' + this.fM(r.valor) + ' \\u00b7 ' + r.peso.toFixed(2) + ' %|P&L ' + this.fM(r.pnl) + ' (' + this.fPct(r.rend) + ')|D\\u00eda ' + this.fPct(r.vd) }))")
    html = _sub(html,
        "secG.map(g => ({ value: g.valor, name: g.name, peso: g.peso }))",
        "secG.map(g => ({ value: g.valor, name: g.name, peso: g.peso, "
        "tip: g.name + '|' + this.fM(g.valor) + ' \\u00b7 ' + g.peso.toFixed(1) + ' %|P&L ' + this.fM(g.pnl) + ' (' + this.fPct(g.rend) + ')|' + g.n + ' posiciones' }))")
    html = _sub(html,
        "emisora: er.emisora, pesoF: er.peso.toFixed(1) + ' %',",
        "emisora: er.emisora, pesoF: er.peso.toFixed(1) + ' %', tip: er.tip,")
    html = _sub(html,
        "const waterfall = wfRaw.map(w => ({ label: w.label, valF: (w.label === 'Bench.' || w.label === 'Port.') ? w.end.toFixed(2) : '+' + (w.end - w.start).toFixed(2), color: w.color, hpx: (Math.abs(w.end - w.start) * scale).toFixed(0), basepx: (Math.min(w.start, w.end) * scale).toFixed(0) }));",
        "const waterfall = wfRaw.map(w => { const dif = w.end - w.start; "
        "const valF = (w.label === 'Bench.' || w.label === 'Port.') ? w.end.toFixed(2) : (dif >= 0 ? '+' : '') + dif.toFixed(2); "
        "return { label: w.label, valF, color: w.color, tip: w.label + '|' + valF + ' pp', "
        "hpx: (Math.abs(dif) * scale).toFixed(0), basepx: (Math.min(w.start, w.end) * scale).toFixed(0) }; });")
    html = _sub(html,
        "const flowBars = flowArr.map(([label, v]) => ({ label: label.replace(' Jul', ''),",
        "const flowBars = flowArr.map(([label, v]) => ({ label: label.replace(' Jul', ''), "
        "tip: label + '|Flujo neto ' + this.fMoney(v, 2),")

    # ---- 3. residuos fijos de la leyenda del template --------------------
    riesgo = d["riesgo"]
    ipc_chart = serie_perf["tips"][-1].split("|")[-1].replace("IPC ", "")
    delta_ipc = float(ipc_chart) - 100.0
    html = _sub(html, ">IPC +7.10 %<", f">IPC {delta_ipc:+.2f} %<")
    html = _sub(html, "Activo +{{ activoF }} pp", "Activo {{ activoF }} pp")
    html = _sub(html, '<span style=\\"color:var(--pos)\\">Activo ',
                '<span style=\\"color:{{ activoCol }}\\">Activo ')
    html = _sub(html, "activoF: activo.toFixed(2),",
                "activoF: (activo >= 0 ? '+' : '') + activo.toFixed(2), "
                "activoCol: this.col(activo),")

    # ---- 4. runtime de tooltip + series para el crosshair ----------------
    series = ("<script>window.__SERIES__ = "
              + _js_plano(dict(perf=serie_perf, dd=serie_dd))
              + ";</script>")
    runtime = """
<script>
(function () {
  var tip = document.createElement('div');
  tip.id = 'bz-tip';
  tip.style.cssText = 'position:fixed;z-index:99999;pointer-events:none;' +
    'display:none;max-width:300px;padding:7px 11px;border-radius:5px;' +
    'font:500 11px/1.5 Inter,-apple-system,sans-serif;' +
    'background:var(--surf2,#1c1c26);color:var(--ink,#f4f3f7);' +
    'border:1px solid var(--border2,#3a3a4d);' +
    'box-shadow:0 4px 16px rgba(0,0,0,.35);white-space:nowrap';
  document.body.appendChild(tip);

  function mostrar(texto, x, y) {
    // El runtime del tablero reconstruye el body al arrancar y se lleva el
    // div; re-adjuntarlo bajo demanda lo hace inmune a cualquier re-render.
    if (!tip.isConnected) document.body.appendChild(tip);
    tip.innerHTML = '';
    texto.split('|').forEach(function (linea, i) {
      var div = document.createElement('div');
      div.textContent = linea.trim();
      if (i === 0) div.style.cssText = 'font-weight:700;margin-bottom:2px';
      tip.appendChild(div);
    });
    tip.style.display = 'block';
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var nx = x + 14, ny = y + 14;
    if (nx + w > innerWidth - 8) nx = x - w - 14;
    if (ny + h > innerHeight - 8) ny = y - h - 14;
    tip.style.left = nx + 'px';
    tip.style.top = ny + 'px';
  }
  function ocultar() { tip.style.display = 'none'; }

  var guia = null;
  function dibujarGuia(svg, S, i) {
    if (!guia || guia.ownerSVGElement !== svg) {
      quitarGuia();
      guia = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      guia.setAttribute('stroke', 'var(--ink3,#7d7a90)');
      guia.setAttribute('stroke-width', '1');
      guia.setAttribute('stroke-dasharray', '2 3');
      guia.setAttribute('pointer-events', 'none');
      svg.appendChild(guia);
    }
    var x = S.padL + i / (S.n - 1) * (S.W - S.padL - S.padR);
    var vb = svg.viewBox.baseVal;
    guia.setAttribute('x1', x); guia.setAttribute('x2', x);
    guia.setAttribute('y1', 0); guia.setAttribute('y2', vb ? vb.height : 300);
  }
  function quitarGuia() {
    if (guia && guia.parentNode) guia.parentNode.removeChild(guia);
    guia = null;
  }

  document.addEventListener('mousemove', function (e) {
    var el = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el && el.getAttribute('data-tip')) {
      quitarGuia();
      mostrar(el.getAttribute('data-tip'), e.clientX, e.clientY);
      return;
    }
    var svg = e.target && e.target.closest ? e.target.closest('svg') : null;
    var pl = svg ? svg.querySelector('[data-chart]') : null;
    if (pl && window.__SERIES__) {
      var S = window.__SERIES__[pl.getAttribute('data-chart')];
      if (S) {
        var r = svg.getBoundingClientRect();
        var fx = (e.clientX - r.left) / r.width * S.W;
        var i = Math.round((fx - S.padL) / (S.W - S.padL - S.padR) * (S.n - 1));
        if (i >= 0 && i < S.n) {
          mostrar(S.tips[i], e.clientX, e.clientY);
          dibujarGuia(svg, S, i);
          return;
        }
      }
    }
    ocultar(); quitarGuia();
  }, true);
  document.addEventListener('mouseleave', function () {
    ocultar(); quitarGuia();
  }, true);
})();
</script>
"""
    cierre = "</body>\n</html>"
    assert html.rstrip().endswith(cierre.replace("\n", "\n")) or cierre in html[-200:], \
        "No se encontro el cierre del documento exterior"
    idx = html.rfind("</body>")
    return html[:idx] + series + runtime + html[idx:]


def _js_plano(v) -> str:
    """
    Serializador para el documento EXTERIOR (sin doble escape): comillas
    simples y \\uXXXX para no-ASCII, apto para un <script> normal.
    """
    if isinstance(v, str):
        cuerpo = "".join(
            c if 32 <= ord(c) < 127 and c not in "'\\<>" else f"\\u{ord(c):04x}"
            for c in v)
        return f"'{cuerpo}'"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v)) if np.isfinite(v) else "0"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_js_plano(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_js_plano(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"No serializable: {type(v)}")


# --------------------------------------------------------------------------
# Tabla de posiciones: rendimiento del dia y acumulado como columnas insignia
# --------------------------------------------------------------------------
# Ambos numeros ya existian pero como texto plano y con encabezados ambiguos
# ("Var. dia", "Rend."). Se convierten en columnas de primera clase: el
# rendimiento del dia como pildora con flecha y fondo tintado por signo, y el
# acumulado como valor mas barra de magnitud (escalada al mayor |rend| de la
# cartera), de modo que el ojo compare emisoras sin leer cada cifra. La fila
# completa gana ficha al pasar el cursor.

def _tabla_posiciones(html: str) -> str:
    # ---- busqueda: el input del template no tenia binding ----------------
    # El filtro busca en emisora, sector, industria y mercado, sin distinguir
    # mayusculas. La barra de magnitud se escala sobre TODAS las filas, no
    # sobre las filtradas, para que no cambie de escala al teclear.
    html = _sub(html,
        "tab: 'resumen', light: false, seg: 'sector', period: '3 meses',",
        "tab: 'resumen', light: false, seg: 'sector', period: '3 meses', filtro: '',")
    html = _sub(html,
        "segTabs, segLabel: segMap[this.state.seg], segBars, segRows,",
        "filtro: this.state.filtro, "
        "setFiltro: e => this.setState({ filtro: e.target.value }), "
        "segTabs, segLabel: segMap[this.state.seg], segBars, segRows,")

    # ---- datos: filtrado + campos nuevos en posRows ----------------------
    html = _sub(html,
        "const posRows = rows.map(r => ({",
        "const filtro = (this.state.filtro || '').trim().toLowerCase(); "
        "const rowsFiltradas = filtro ? rows.filter(r => "
        "(r.emisora + ' ' + r.sector + ' ' + r.industria + ' ' + r.mercado)"
        ".toLowerCase().includes(filtro)) : rows; "
        "const maxAbsRend = Math.max(...rows.map(r => Math.abs(r.rend))) || 1; "
        "const posRows = rowsFiltradas.map(r => ({")
    html = _sub(html,
        "varDia: this.fPct(r.vd), varColor: this.col(r.vd),",
        "varDia: this.fPct(r.vd), varColor: this.col(r.vd), "
        "vdArrow: this.arrow(r.vd), "
        "vdBg: r.vd > 0 ? 'color-mix(in srgb, var(--pos) 13%, transparent)' "
        ": r.vd < 0 ? 'color-mix(in srgb, var(--neg) 13%, transparent)' : 'var(--surf3)', "
        "tipFila: r.emisora + '|Rend. d\\u00eda ' + this.fPct(r.vd) + ' \\u00b7 ' + this.fMoney(r.pnlDia) "
        "+ '|Rend. acumulado ' + this.fPct(r.rend) + ' \\u00b7 P&L ' + this.fMoney(r.pnl) "
        "+ '|Peso ' + r.peso.toFixed(2) + ' %',")
    html = _sub(html,
        "rend: this.fPct(r.rend), rendColor: this.col(r.rend),",
        "rend: this.fPct(r.rend), rendColor: this.col(r.rend), "
        "rendW: (Math.abs(r.rend) / maxAbsRend * 100).toFixed(1), "
        "rendBarColor: r.rend >= 0 ? 'var(--pos)' : 'var(--neg)',")

    # ---- encabezados explicitos -----------------------------------------
    html = _sub(html, 'padding:9px 10px\\">Var. día<',
                'padding:9px 10px\\">Rend. día<')
    html = _sub(html, 'padding:9px 10px\\">Rend.<',
                'padding:9px 10px\\">Rend. acumulado<')

    # ---- celda del dia: pildora con flecha y fondo por signo -------------
    html = _sub(html,
        '<sc-raw-td style=\\"padding:7px 10px;text-align:right;font-family:var(--mono);'
        'color:{{ r.varColor }}\\">{{ r.varDia }}<\\u002Fsc-raw-td>',
        '<sc-raw-td style=\\"padding:5px 10px;text-align:right\\">'
        '<span style=\\"display:inline-block;min-width:76px;text-align:right;'
        'padding:3px 8px;border-radius:4px;background:{{ r.vdBg }};'
        'color:{{ r.varColor }};font-family:var(--mono);font-weight:600\\">'
        '{{ r.vdArrow }}{{ r.varDia }}<\\u002Fspan><\\u002Fsc-raw-td>')

    # ---- celda del acumulado: valor + barra de magnitud ------------------
    html = _sub(html,
        '<sc-raw-td style=\\"padding:7px 10px;text-align:right;font-family:var(--mono);'
        'color:{{ r.rendColor }}\\">{{ r.rend }}<\\u002Fsc-raw-td>',
        '<sc-raw-td style=\\"padding:5px 10px;text-align:right\\">'
        '<div style=\\"display:flex;align-items:center;justify-content:flex-end;gap:7px\\">'
        '<div style=\\"width:46px;height:5px;background:var(--surf3);'
        'border-radius:3px;overflow:hidden\\">'
        '<div style=\\"width:{{ r.rendW }}%;height:100%;background:{{ r.rendBarColor }};'
        'border-radius:3px\\"><\\u002Fdiv><\\u002Fdiv>'
        '<span style=\\"font-family:var(--mono);font-weight:600;min-width:64px;'
        'color:{{ r.rendColor }}\\">{{ r.rend }}<\\u002Fspan>'
        '<\\u002Fdiv><\\u002Fsc-raw-td>')

    # ---- binding del input de busqueda -----------------------------------
    html = _sub(html,
        '<input placeholder=\\"Filtrar emisora…\\" style=',
        '<input placeholder=\\"Filtrar emisora…\\" value=\\"{{ filtro }}\\" '
        'sc-camel-on-input=\\"{{ setFiltro }}\\" '
        'sc-camel-on-change=\\"{{ setFiltro }}\\" style=')

    # ---- ficha de la fila al pasar el cursor -----------------------------
    html = _sub(html,
        'list=\\"{{ posRows }}\\" as=\\"r\\" hint-placeholder-count=\\"14\\">\\n'
        '            <sc-raw-tr style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">',
        'list=\\"{{ posRows }}\\" as=\\"r\\" hint-placeholder-count=\\"14\\">\\n'
        '            <sc-raw-tr data-tip=\\"{{ r.tipFila }}\\" '
        'style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">')
    return html


# --------------------------------------------------------------------------
# Riesgo explicativo: eje y marcadores en el histograma del VaR, y notas de
# lectura con datos reales bajo cada grafica de la pestana
# --------------------------------------------------------------------------
# Las graficas de riesgo eran marcas sin apoyo: el histograma no tenia eje ni
# senalaba donde caen los VaR, y ninguna grafica decia como leerse. Los
# textos viven en risk.* (se calculan en _bloque_riesgo con las cifras del
# mandato); aqui solo se les da lugar en el template. Debe ejecutarse ANTES
# de _capa_interactiva para trabajar sobre anclas pristinas.

_NOTA_STL = ('font:400 10.5px/1.55 var(--sans);color:var(--ink3);'
             'margin:0 0 10px')


def _nota(campo: str) -> str:
    return ('<div style=\\"' + _NOTA_STL + '\\">{{ risk.' + campo
            + ' }}<\\u002Fdiv>')


def _riesgo_explicativo(html: str) -> str:
    # ---- 1. histograma: contenedor relativo con espacio para etiquetas ---
    html = _sub(html,
        '<div style=\\"display:flex;align-items:flex-end;gap:2px;'
        'height:160px;padding-top:8px\\">',
        '<div style=\\"position:relative;display:flex;align-items:flex-end;'
        'gap:2px;height:160px;padding-top:8px;margin-top:18px\\">')

    # ---- 2. marcadores de VaR 95/99 sobre la distribucion ----------------
    html = _sub(html,
        'min-height:2px\\"><\\u002Fdiv><\\u002Fsc-for>',
        'min-height:2px\\"><\\u002Fdiv><\\u002Fsc-for>'
        '<div data-tip=\\"{{ risk.var95Tip }}\\" style=\\"position:absolute;'
        'left:{{ risk.var95X }}%;top:0;bottom:0;'
        'border-left:2px dashed var(--neg);opacity:.7\\"><\\u002Fdiv>'
        '<div style=\\"position:absolute;left:{{ risk.var95X }}%;top:-15px;'
        'transform:translateX(-50%);font:700 8.5px var(--mono);'
        'color:var(--neg);white-space:nowrap;opacity:.85\\">VaR 95'
        '<\\u002Fdiv>'
        '<div data-tip=\\"{{ risk.var99Tip }}\\" style=\\"position:absolute;'
        'left:{{ risk.var99X }}%;top:0;bottom:0;'
        'border-left:2px solid var(--neg)\\"><\\u002Fdiv>'
        '<div style=\\"position:absolute;left:{{ risk.var99X }}%;top:-15px;'
        'transform:translateX(-50%);font:700 8.5px var(--mono);'
        'color:var(--neg);white-space:nowrap\\">VaR 99<\\u002Fdiv>')

    # ---- 3. eje X + leyenda de zonas + nota de lectura -------------------
    html = _sub(html,
        '<div style=\\"display:flex;gap:16px;margin-top:10px;'
        'font-family:var(--mono);font-size:10px\\">'
        '<span style=\\"color:var(--neg)\\">VaR 95 · −1.62 %'
        '<\\u002Fspan>'
        '<span style=\\"color:var(--neg)\\">VaR 99 · −2.74 %'
        '<\\u002Fspan>'
        '<span style=\\"color:var(--ink3)\\">CVaR 95 · −2.19 %'
        '<\\u002Fspan><\\u002Fdiv>',
        # eje X
        '<div style=\\"display:flex;justify-content:space-between;'
        'font:500 9px var(--mono);color:var(--ink3);margin-top:3px\\">'
        '<span>{{ risk.ejeLo }}<\\u002Fspan>'
        '<span>{{ risk.ejeMedLo }}<\\u002Fspan><span>0<\\u002Fspan>'
        '<span>{{ risk.ejeMedHi }}<\\u002Fspan>'
        '<span>{{ risk.ejeHi }}<\\u002Fspan><\\u002Fdiv>'
        # leyenda de zonas
        '<div style=\\"display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;'
        'font-family:var(--mono);font-size:10px;align-items:center\\">'
        '<span style=\\"display:inline-flex;align-items:center;gap:5px\\">'
        '<span style=\\"width:9px;height:9px;border-radius:2px;'
        'background:var(--brand-lite);display:inline-block\\"><\\u002Fspan>'
        'sesión típica<\\u002Fspan>'
        '<span style=\\"display:inline-flex;align-items:center;gap:5px\\">'
        '<span style=\\"width:9px;height:9px;border-radius:2px;'
        'background:color-mix(in srgb, var(--neg) 55%, transparent);'
        'display:inline-block\\"><\\u002Fspan>'
        'cola 5 % · VaR 95 {{ risk.var95F }}<\\u002Fspan>'
        '<span style=\\"display:inline-flex;align-items:center;gap:5px\\">'
        '<span style=\\"width:9px;height:9px;border-radius:2px;'
        'background:var(--neg);display:inline-block\\"><\\u002Fspan>'
        'cola 1 % · VaR 99 {{ risk.var99F }}<\\u002Fspan>'
        '<span style=\\"color:var(--ink3)\\">CVaR 95 {{ risk.cvarF }}'
        '<\\u002Fspan><\\u002Fdiv>'
        # nota de lectura
        '<div style=\\"font:400 10.5px/1.6 var(--sans);'
        'color:var(--ink3);margin-top:9px;border-top:1px dashed var(--border);'
        'padding-top:8px\\">{{ risk.notaVar }}<\\u002Fdiv>')

    # ---- 4. notas de lectura en el resto de las graficas -----------------
    for campo, ancla in [
        ("notaStress",
         '<sc-for list=\\"{{ risk.stress }}\\" as=\\"s\\" '
         'hint-placeholder-count=\\"7\\">'),
        ("notaFactores",
         '<sc-for list=\\"{{ risk.factors }}\\" as=\\"f\\" '
         'hint-placeholder-count=\\"8\\">'),
        ("notaPesoRiesgo",
         '<sc-for list=\\"{{ riskWeight }}\\" as=\\"r\\" '
         'hint-placeholder-count=\\"8\\">'),
        ("notaCorr",
         '<sc-for list=\\"{{ corrRows }}\\" as=\\"row\\" '
         'hint-placeholder-count=\\"8\\">'),
    ]:
        html = _sub(html, ancla, _nota(campo) + ancla)

    # ---- 5. dispersion: ejes con nombre completo + nota ------------------
    html = _sub(html, '>vol →<', '>Volatilidad anual (%) →<')
    html = _sub(html, '>rend<', '>↑ Rend. (%)<')
    html = _sub(html,
        '>↑ Rend. (%)<\\u002Ftext><\\u002Fsvg>',
        '>↑ Rend. (%)<\\u002Ftext><\\u002Fsvg>' + _nota("notaScatter"))

    # ---- 6. drawdown: nota tras su grafica -------------------------------
    rx = re.compile(
        re.escape('points=\\"{{ risk.ddLine }}\\"') + ".*?"
        + re.escape('<\\u002Fsvg>'), re.S)
    nuevo, cuantos = rx.subn(lambda m: m.group(0) + _nota("notaDd"), html,
                             count=1)
    assert cuantos == 1, "No se encontro la grafica de drawdown"
    return nuevo


def html_con_datos_reales(ruta_html: Path | None = None) -> str:
    ruta = ruta_html or (RAIZ / "assets" / "dashboard_baz.html")
    html = ruta.read_text(encoding="utf-8")
    return inyectar(html, calcular())
