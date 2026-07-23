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

    # --- 7a. pestana de rotacion sectorial --------------------------------
    html = _seccion_rotacion(html, _rotacion(d))

    # --- 7a2. pestana de deuda gubernamental (vector Valmer) --------------
    deu = _deuda()
    serie_curva = None
    if deu is not None:
        serie_curva = deu.pop("serieCurva", None)
        serie_udinom = deu.pop("serieUdinom", None)
        html = _seccion_deuda(html, deu)
        pc = _cartera_rf()
        if pc is not None:
            html = _seccion_cartera(html, pc)

    # --- 7b. riesgo explicativo (antes de la capa interactiva) ------------
    html = _riesgo_explicativo(html)

    # --- 8. capa interactiva: tooltips y crosshair ------------------------
    html = _capa_interactiva(html, d, serie_perf, serie_dd, serie_curva,
                             serie_udinom)
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
                      serie_dd: dict,
                      serie_curva: dict | None = None,
                      serie_udinom: dict | None = None) -> str:
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
              + _js_plano(dict(perf=serie_perf, dd=serie_dd,
                               **({"curva": serie_curva}
                                  if serie_curva else {}),
                               **({"udinom": serie_udinom}
                                  if serie_udinom else {})))
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


# --------------------------------------------------------------------------
# Rotacion sectorial y sugerencias de ponderacion
# --------------------------------------------------------------------------
# Mapa de rotacion al estilo RRG simplificado: eje X = fuerza relativa del
# sector contra el IPC a 3 meses; eje Y = fuerza relativa a 1 mes. Cuatro
# cuadrantes: Lider (+/+), Debilitandose (+/-), Rezagado (-/-) y Mejorando
# (-/+). Las sugerencias de ponderacion salen de reglas transparentes que
# cruzan el cuadrante con el peso activo (portafolio - benchmark), la
# seleccion de Brinson a 3 meses y el consumo de riesgo; son un insumo de
# analisis, no una recomendacion de inversion, y la pestana lo declara.

_ABREV_SECTOR = {
    "Consumo Básico": "C. Básico", "Consumo Discrecional": "C. Discr.",
    "Índice Amplio": "Í. Amplio", "Telecomunicaciones": "Telecom",
    "Inmobiliario": "Inmob.", "Industriales": "Industr.",
    "Financiero": "Financ.", "Materiales": "Mater.",
    "Tecnología": "Tecnol.", "Liquidez": "Liquidez", "Salud": "Salud",
}

_CUADRANTE_COLOR = {
    "Líder": "var(--pos)", "Mejorando": "var(--brand-lite)",
    "Debilitándose": "var(--warn)", "Rezagado": "var(--neg)",
}
_SUGERENCIA_COLOR = {
    "Aumentar": "var(--pos)", "Acumular": "var(--brand-lite)",
    "Vigilar": "var(--warn)", "Reducir": "var(--neg)",
    "Revisar selección": "var(--warn)", "Mantener": "var(--ink3)",
}


def _rend_sector_portafolio(historico, val, sector_ascii: str,
                            ses: int) -> float | None:
    """Rendimiento del sector ponderado por valor con las emisoras propias."""
    sub = val[val["sector"] == sector_ascii]
    piezas = []
    for _, r_ in sub.iterrows():
        rr = bmk.rendimiento_periodo(historico, r_["ticker"], ses)
        if rr is not None:
            piezas.append((float(r_["valor_mercado"]), rr))
    if not piezas:
        return None
    total = sum(w for w, _ in piezas)
    return sum(w * rr for w, rr in piezas) / total


def _rotacion(d: dict) -> dict:
    historico, bench_df, val = d["historico"], d["bench_df"], d["val"]
    ventanas = {"1m": 21, "3m": 63, "6m": 126}

    ipc = {k: bmk.rendimiento_periodo(historico, BENCHMARK_TICKER, s)
           for k, s in ventanas.items()}
    # Todo en nombres UI (con acento): el CSV del benchmark y la taxonomia
    # interna van en ASCII y sin mapear duplicaban sectores.
    bench_sec = {k: {_ui(SECTOR_UI, f_.sector): f_.rendimiento_pct / 100.0
                     for f_ in bmk.sectores_benchmark(
                         bench_df, historico, s).itertuples()}
                 for k, s in ventanas.items()}

    w_p_raw = (val.groupby("sector")["valor_mercado"].sum()
               / val["valor_mercado"].sum() * 100.0)
    w_p = {_ui(SECTOR_UI, k): float(v) for k, v in w_p_raw.items()}
    w_b = {_ui(SECTOR_UI, k): float(v) for k, v in
           bench_df.groupby("sector")["peso_pct"].sum().items()}
    ascii_de = {_ui(SECTOR_UI, k): k for k in w_p_raw.index}

    # seleccion Brinson a 3 meses por sector
    bs63 = bmk.sectores_benchmark(bench_df, historico, 63)
    bf63 = an.brinson_fachler(val, historico, bs63, 63)
    sel3 = {_ui(SECTOR_UI, t.sector): float(t.seleccion_pp)
            for t in bf63["tabla"].itertuples()}

    sectores = sorted(set(w_p) | set(w_b))
    filas, puntos = [], []
    for sec in sectores:
        rends = {}
        for k, s in ventanas.items():
            r_ = bench_sec[k].get(sec)
            if r_ is None and sec in ascii_de:
                r_ = _rend_sector_portafolio(historico, val, ascii_de[sec], s)
            rends[k] = r_
        if rends["3m"] is None or rends["1m"] is None:
            continue

        fr1 = (rends["1m"] - ipc["1m"]) * 100.0
        fr3 = (rends["3m"] - ipc["3m"]) * 100.0
        cuad = ("Líder" if fr3 >= 0 and fr1 >= 0 else
                "Debilitándose" if fr3 >= 0 else
                "Mejorando" if fr1 >= 0 else "Rezagado")

        wp, wb = w_p.get(sec, 0.0), w_b.get(sec, 0.0)
        act = wp - wb
        sl = sel3.get(sec)
        fuera = wb == 0.0

        # reglas de sugerencia (umbral de peso activo: 2 pp)
        if cuad == "Líder" and act < -2:
            sug, razon = "Aumentar", (
                f"momentum relativo positivo (FR 3m {fr3:+.1f} pp) con "
                f"subponderación de {-act:.1f} pp")
        elif cuad == "Mejorando" and act < -2:
            sug, razon = "Acumular", (
                f"el sector acelera (FR 1m {fr1:+.1f} pp) desde rezago; "
                f"subponderación de {-act:.1f} pp")
        elif cuad == "Rezagado" and act > 2:
            sug, razon = "Reducir", (
                f"momentum relativo negativo (FR 3m {fr3:+.1f} pp) con "
                f"sobreponderación de {act:.1f} pp")
        elif cuad == "Debilitándose" and act > 2:
            sug, razon = "Vigilar", (
                f"el sector pierde tracción (FR 1m {fr1:+.1f} pp) y se "
                f"sobrepondera {act:.1f} pp")
        else:
            sug, razon = "Mantener", "peso activo dentro de banda (±2 pp)"
        if sl is not None and sl < -0.3:
            if sug == "Mantener":
                sug = "Revisar selección"
                razon = f"la selección de emisoras resta {abs(sl):.1f} pp a 3m"
            else:
                razon += f"; la selección resta {abs(sl):.1f} pp"
        if fuera:
            razon += " · fuera del índice"

        def _pc(v, dec=1):
            return (f"{v*100:+.{dec}f} %" if v is not None else "—")

        filas.append(dict(
            sector=sec, wpF=f"{wp:.1f} %", wbF=f"{wb:.1f} %",
            actF=f"{act:+.1f} pp",
            actC="var(--pos)" if act > 0 else "var(--neg)" if act < 0 else "var(--ink3)",
            r1F=_pc(rends["1m"]),
            r1C="var(--pos)" if rends["1m"] >= 0 else "var(--neg)",
            r3F=_pc(rends["3m"]),
            r3C="var(--pos)" if rends["3m"] >= 0 else "var(--neg)",
            r6F=_pc(rends["6m"]),
            r6C=("var(--pos)" if (rends["6m"] or 0) >= 0 else "var(--neg)"),
            frF=f"{fr3:+.1f} pp",
            frC="var(--pos)" if fr3 >= 0 else "var(--neg)",
            cuad=cuad, cuadC=_CUADRANTE_COLOR[cuad],
            sug=sug, sugC=_SUGERENCIA_COLOR[sug],
            tip=(f"{sec}|{cuad} · FR 3m {fr3:+.1f} pp · FR 1m {fr1:+.1f} pp"
                 f"|Peso {wp:.1f} % vs bench {wb:.1f} % ({act:+.1f} pp)"
                 f"|{sug}: {razon}"),
            _orden=(0 if sug != "Mantener" else 1, -abs(act)),
            _fr1=fr1, _fr3=fr3, _wp=wp, _razon=razon,
        ))

    # burbujas del mapa (viewBox 520 x 290)
    W_, H_, mx_, my_ = 520.0, 290.0, 46.0, 30.0
    lim_x = max(abs(f["_fr3"]) for f in filas) * 1.2 or 1.0
    lim_y = max(abs(f["_fr1"]) for f in filas) * 1.2 or 1.0
    for f in filas:
        cx = (W_ / 2) + f["_fr3"] / lim_x * (W_ / 2 - mx_)
        cy = (H_ / 2) - f["_fr1"] / lim_y * (H_ / 2 - my_)
        rr = 5.0 + (f["_wp"] ** 0.5) * 2.1
        puntos.append(dict(
            cx=f"{cx:.0f}", cy=f"{cy:.0f}", r=f"{rr:.1f}",
            color=f["cuadC"], label=_ABREV_SECTOR.get(f["sector"], f["sector"]),
            lx=f"{cx:.0f}", ly=f"{cy - rr - 5:.0f}", tip=f["tip"]))

    filas.sort(key=lambda f: f["_orden"])

    # sugerencias destacadas (las accionables primero)
    iconos = {"Aumentar": ("▲", "var(--pos)"), "Acumular": ("▲", "var(--brand-lite)"),
              "Reducir": ("▼", "var(--neg)"), "Vigilar": ("◆", "var(--warn)"),
              "Revisar selección": ("✎", "var(--warn)")}
    sugerencias = []
    for f in filas:
        if f["sug"] == "Mantener":
            continue
        ic, col = iconos.get(f["sug"], ("•", "var(--ink2)"))
        sugerencias.append(dict(
            icon=ic, color=col,
            titulo=f"{f['sector']} — {f['sug'].lower()}",
            detalle=f["_razon"][0].upper() + f["_razon"][1:] + "."))
    if not sugerencias:
        sugerencias = [dict(icon="✓", color="var(--pos)",
                            titulo="Ponderación alineada",
                            detalle="Ningún sector fuera de banda con las "
                                    "reglas vigentes (±2 pp de peso activo).")]

    for f in filas:
        for k in ("_orden", "_fr1", "_fr3", "_wp", "_razon"):
            f.pop(k)

    return dict(
        filas=filas, puntos=puntos, sugerencias=sugerencias[:5],
        ipc1F=f"{ipc['1m']*100:+.1f} %", ipc3F=f"{ipc['3m']*100:+.1f} %",
        ipc6F=(f"{ipc['6m']*100:+.1f} %" if ipc["6m"] is not None else "—"),
        notaMapa=("Cada burbuja es un sector; el tamaño es su peso en el "
                  "portafolio y el color su cuadrante. La fuerza relativa "
                  "(FR) es el rendimiento del sector menos el del IPC en la "
                  "ventana. El giro típico es horario: Mejorando → Líder → "
                  "Debilitándose → Rezagado."),
        notaMetodo=("Rendimientos sectoriales del benchmark reconstruido a "
                    "nivel constituyente (sectores fuera del índice: con las "
                    "emisoras del portafolio). Sugerencias por reglas "
                    "cuantitativas: cuadrante de rotación × peso activo "
                    "(banda ±2 pp) × selección Brinson 3m. Documento de "
                    "trabajo; no constituye una recomendación de inversión."),
    )


def _seccion_rotacion(html: str, rot: dict) -> str:
    # ---- navegacion, titulos y bandera de pestana ------------------------
    html = _sub(html,
        "['diagnostico','Diagnóstico','08'],['configuracion','Configuración','09']",
        "['diagnostico','Diagnóstico','08'],['rotacion','Rotación','09'],"
        "['configuracion','Configuración','10']")
    html = _sub(html,
        "configuracion: ['Configuración', 'Fuentes de datos",
        "rotacion: ['Rotación sectorial', 'Momentum relativo vs IPC y "
        "sugerencias de ponderación'], configuracion: ['Configuración', "
        "'Fuentes de datos")
    html = _sub(html,
        "isOperaciones: this.state.tab === 'operaciones', isDiagnostico: "
        "this.state.tab === 'diagnostico', isConfiguracion: this.state.tab "
        "=== 'configuracion',",
        "isOperaciones: this.state.tab === 'operaciones', isDiagnostico: "
        "this.state.tab === 'diagnostico', isConfiguracion: this.state.tab "
        "=== 'configuracion', isRotacion: this.state.tab === 'rotacion', "
        "rot: " + js(rot) + ",")

    # ---- seccion del template -------------------------------------------
    TD = '<\\u002Fdiv>'
    TS = '<\\u002Fspan>'
    card = ('background:var(--surf);border:1px solid var(--border);'
            'border-radius:6px;padding:14px 16px')
    header = ('font:700 10px/1 var(--sans);letter-spacing:.14em;'
              'text-transform:uppercase;color:var(--ink3);margin-bottom:12px')
    nota = ('font:400 10.5px/1.55 var(--sans);color:var(--ink3);'
            'margin-top:10px;border-top:1px dashed var(--border);'
            'padding-top:8px')
    th = ('text-align:right;font:700 9px/1 var(--sans);'
          'letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);'
          'padding:9px 10px')
    td = 'padding:7px 10px;text-align:right;font-family:var(--mono)'

    seccion = (
        '<!-- ================= ROTACIÓN ================= -->\\n'
        '      <sc-if value=\\"{{ isRotacion }}\\" '
        'hint-placeholder-val=\\"{{ false }}\\">\\n'
        '      <div style=\\"display:grid;grid-template-columns:1.25fr 1fr;'
        'gap:14px;margin-bottom:14px\\">\\n'
        # ---- mapa de rotacion ----
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Mapa de rotación sectorial'
        '<span style=\\"font-weight:500;text-transform:none;letter-spacing:0\\">'
        ' · FR = sector − IPC' + TS + TD +
        '<svg viewBox=\\"0 0 520 290\\" style=\\"width:100%;height:auto\\">'
        '<rect x=\\"0\\" y=\\"0\\" width=\\"520\\" height=\\"290\\" '
        'fill=\\"var(--surf2)\\" rx=\\"4\\"><\\u002Frect>'
        '<line x1=\\"260\\" y1=\\"8\\" x2=\\"260\\" y2=\\"282\\" '
        'stroke=\\"var(--border2)\\"><\\u002Fline>'
        '<line x1=\\"10\\" y1=\\"145\\" x2=\\"510\\" y2=\\"145\\" '
        'stroke=\\"var(--border2)\\"><\\u002Fline>'
        '<text x=\\"14\\" y=\\"20\\" font-family=\\"var(--sans)\\" '
        'font-size=\\"10\\" font-weight=\\"700\\" '
        'fill=\\"var(--brand-lite)\\" opacity=\\".8\\">MEJORANDO<\\u002Ftext>'
        '<text x=\\"506\\" y=\\"20\\" text-anchor=\\"end\\" '
        'font-family=\\"var(--sans)\\" font-size=\\"10\\" '
        'font-weight=\\"700\\" fill=\\"var(--pos)\\" opacity=\\".8\\">'
        'LÍDER<\\u002Ftext>'
        '<text x=\\"14\\" y=\\"280\\" font-family=\\"var(--sans)\\" '
        'font-size=\\"10\\" font-weight=\\"700\\" fill=\\"var(--neg)\\" '
        'opacity=\\".8\\">REZAGADO<\\u002Ftext>'
        '<text x=\\"506\\" y=\\"280\\" text-anchor=\\"end\\" '
        'font-family=\\"var(--sans)\\" font-size=\\"10\\" '
        'font-weight=\\"700\\" fill=\\"var(--warn)\\" opacity=\\".8\\">'
        'DEBILITÁNDOSE<\\u002Ftext>'
        '<text x=\\"506\\" y=\\"158\\" text-anchor=\\"end\\" '
        'font-family=\\"var(--mono)\\" font-size=\\"8.5\\" '
        'fill=\\"var(--ink3)\\">FR 3m →<\\u002Ftext>'
        '<text x=\\"266\\" y=\\"18\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"8.5\\" fill=\\"var(--ink3)\\">↑ FR 1m<\\u002Ftext>'
        '<sc-for list=\\"{{ rot.puntos }}\\" as=\\"p\\" '
        'hint-placeholder-count=\\"8\\">'
        '<circle data-tip=\\"{{ p.tip }}\\" cx=\\"{{ p.cx }}\\" '
        'cy=\\"{{ p.cy }}\\" r=\\"{{ p.r }}\\" fill=\\"{{ p.color }}\\" '
        'opacity=\\"0.78\\" stroke=\\"var(--surf)\\" '
        'stroke-width=\\"1.5\\"><\\u002Fcircle>'
        '<text x=\\"{{ p.lx }}\\" y=\\"{{ p.ly }}\\" '
        'text-anchor=\\"middle\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"8.5\\" fill=\\"var(--ink2)\\">{{ p.label }}'
        '<\\u002Ftext><\\u002Fsc-for><\\u002Fsvg>'
        '<div style=\\"' + nota + '\\">{{ rot.notaMapa }}' + TD + TD + '\\n'
        # ---- sugerencias del modelo ----
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Sugerencias de ponderación'
        '<span style=\\"font-weight:500;text-transform:none;letter-spacing:0\\">'
        ' · IPC 1m {{ rot.ipc1F }} · 3m {{ rot.ipc3F }} · 6m {{ rot.ipc6F }}'
        + TS + TD +
        '<sc-for list=\\"{{ rot.sugerencias }}\\" as=\\"g\\" '
        'hint-placeholder-count=\\"4\\">'
        '<div style=\\"display:flex;gap:10px;padding:9px 10px;'
        'margin-bottom:7px;background:var(--surf2);border-radius:5px;'
        'border-left:3px solid {{ g.color }}\\">'
        '<div style=\\"color:{{ g.color }};font-size:12px;line-height:1.3\\">'
        '{{ g.icon }}' + TD +
        '<div><div style=\\"font:600 11.5px/1.3 var(--sans);'
        'color:var(--ink)\\">{{ g.titulo }}' + TD +
        '<div style=\\"font:400 10.5px/1.5 var(--sans);'
        'color:var(--ink2);margin-top:2px\\">{{ g.detalle }}' + TD + TD + TD +
        '<\\u002Fsc-for>'
        '<div style=\\"' + nota + '\\">{{ rot.notaMetodo }}' + TD + TD + TD + '\\n'
        # ---- tabla ----
        '      <div style=\\"' + card.replace('padding:14px 16px',
                                              'padding:0;overflow:hidden')
        + '\\">\\n'
        '        <sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:11px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th.replace('text-align:right', 'text-align:left')
        + '\\">Sector<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">W port.<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">W bench.<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Activo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">1m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">3m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">6m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">FR 3m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th.replace('text-align:right', 'text-align:left')
        + '\\">Cuadrante<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th.replace('text-align:right', 'text-align:left')
        + '\\">Sugerencia<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ rot.filas }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"10\\">'
        '<sc-raw-tr data-tip=\\"{{ f.tip }}\\" '
        'style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:7px 10px;font:600 11px var(--sans);'
        'color:var(--ink)\\">{{ f.sector }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">{{ f.wpF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink3)\\">{{ f.wbF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:{{ f.actC }}\\">{{ f.actF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:{{ f.r1C }}\\">{{ f.r1F }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:{{ f.r3C }}\\">{{ f.r3F }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:{{ f.r6C }}\\">{{ f.r6F }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;color:{{ f.frC }}\\">'
        '{{ f.frF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"padding:5px 10px\\">'
        '<span style=\\"display:inline-block;padding:2.5px 8px;'
        'border-radius:4px;font:600 10px var(--sans);color:{{ f.cuadC }};'
        'background:color-mix(in srgb, currentColor 13%, transparent)\\">'
        '{{ f.cuad }}' + TS + '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"padding:5px 10px\\">'
        '<span style=\\"display:inline-block;padding:2.5px 8px;'
        'border-radius:4px;font:600 10px var(--sans);color:{{ f.sugC }};'
        'background:color-mix(in srgb, currentColor 13%, transparent)\\">'
        '{{ f.sug }}' + TS + '<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        '      <\\u002Fsc-if>\\n\\n      ')

    html = _sub(html,
        '<!-- ================= CONFIGURACIÓN ================= -->',
        seccion + '<!-- ================= CONFIGURACIÓN ================= -->')
    return html


# --------------------------------------------------------------------------
# Renta fija: tres pestanas sobre el vector Valmer
# --------------------------------------------------------------------------
# La seccion de deuda se separa en tres vistas para que datos, analisis y
# valuacion no se mezclen:
#   Mercado    -> lo que DICE el vector: KPIs, curva anotada, sobretasas de
#                 BONDES F, niveles por bono e historico con percentiles.
#   Analisis   -> lo que la mesa OPINA y las herramientas de decision: tesis
#                 autorada, valor relativo, matriz de escenarios, carry y
#                 fijo contra flotante.
#   Valuacion  -> como se CALCULA y el credito: verificacion solver vs tasa
#                 oficial, convenciones, UDI y curvas de sobretasa
#                 corporativa por calificacion.
# La navegacion se agrupa (Renta variable / Renta fija / General) sin tocar
# el marcado: los encabezados son botones inertes con estilo propio.

from . import bonos as bn


def _deuda() -> dict | None:
    dv = bn.cargar()
    if dv is None:
        return None
    a = bn.analitica(dv)
    cr = bn.carry_rolldown(dv)
    res = bn.residuos_curva(dv)
    fwds = bn.forwards_clave(dv)
    mx = bn.matriz_escenarios(dv)
    fondeo = float(cr["fondeo"].iloc[0])
    cete1a = a["cete364"]
    res_por_serie = dict(zip(res["serie"], res["residuo_pb"]))

    # ================= MERCADO =================
    # ---- curva anotada (identica a la version previa) --------------------
    W_, H_, mL, mR, mT, mB = 980.0, 380.0, 50.0, 18.0, 30.0, 34.0
    t_max = float(dv["bonos_m"]["anios"].max()) * 1.03
    y_lo = float(np.floor(dv["udibonos"]["ytm"].min() * 2) / 2 - 0.5)
    y_hi = float(np.ceil(dv["bonos_m"]["ytm"].max() * 2) / 2 + 0.5)

    def X(t):
        return mL + (max(t, 0.0) / t_max) ** 0.5 * (W_ - mL - mR)

    def Y(y):
        return mT + (y_hi - y) / (y_hi - y_lo) * (H_ - mT - mB)

    def linea(df_):
        return " ".join(f"{X(t):.1f},{Y(y):.1f}"
                        for t, y in zip(df_["anios"], df_["ytm"]))

    g_may, g_men = [], []
    v = y_lo
    while v <= y_hi + 1e-9:
        (g_may if abs(v - round(v)) < 0.01 else g_men).append(
            dict(y=f"{Y(v):.1f}", label=f"{v:.0f}%"))
        v += 0.5
    ticks_x = [dict(x=f"{X(t):.1f}", label=lab)
               for t, lab in [(0.083, "1m"), (0.25, "3m"), (0.5, "6m"),
                              (1, "1a"), (2, "2a"), (3, "3a"), (5, "5a"),
                              (7, "7a"), (10, "10a"), (15, "15a"),
                              (20, "20a"), (28.5, "30a")] if t <= t_max]

    puntos, etiquetas_m = [], []
    for i, r_ in enumerate(dv["bonos_m"].itertuples()):
        serie = str(r_.SERIE)
        rpb = res_por_serie.get(serie, 0.0)
        color = ("var(--pos)" if rpb > 4 else
                 "var(--neg)" if rpb < -4 else "var(--brand-lite)")
        estado = ("barato" if rpb > 4 else "caro" if rpb < -4 else "en línea")
        cx, cy = X(r_.anios), Y(r_.ytm)
        puntos.append(dict(
            cx=f"{cx:.1f}", cy=f"{cy:.1f}", r="4.2", color=color,
            tip=(f"BONO M {serie}|Tasa {r_.ytm:.2f} % · "
                 f"{r_.anios:.1f}a · dur {r_.DURACION:.2f}"
                 f"|Valor relativo: {rpb:+.0f} pb ({estado})")))
        etiquetas_m.append(dict(
            x=f"{cx:.1f}",
            y=f"{cy - 11:.1f}" if i % 2 == 0 else f"{cy + 19:.1f}",
            label=serie))
    for r_ in dv["cetes"].itertuples():
        puntos.append(dict(
            cx=f"{X(r_.anios):.1f}", cy=f"{Y(r_.ytm):.1f}", r="2.6",
            color="var(--warn)",
            tip=f"CETE {r_.SERIE}|{int(r_.dias)} días · {r_.ytm:.2f} %"))
    for r_ in dv["udibonos"].itertuples():
        puntos.append(dict(
            cx=f"{X(r_.anios):.1f}", cy=f"{Y(r_.ytm):.1f}", r="3.4",
            color="var(--pos)",
            tip=(f"UDIBONO {r_.SERIE}|Tasa real {r_.ytm:.2f} % · "
                 f"{r_.anios:.1f}a")))

    curva = dict(
        lineaM=linea(dv["bonos_m"]), lineaC=linea(dv["cetes"]),
        lineaS=linea(dv["udibonos"]), puntos=puntos,
        etiquetasM=etiquetas_m, gMay=g_may, gMen=g_men, ticksX=ticks_x,
        fondeoY=f"{Y(fondeo):.1f}", fondeoF=f"Fondeo {fondeo:.2f} %")

    an_, yt_ = bn.curva_nominal(dv)
    an_s = dv["udibonos"]["anios"].values
    yt_s = dv["udibonos"]["ytm"].values
    t_real_min = float(an_s.min())
    n_cx = 90
    tips_curva = []
    for i in range(n_cx):
        t = ((i / (n_cx - 1)) ** 2) * t_max
        nom = float(np.interp(t, an_, yt_))
        if t >= t_real_min:
            re_ = float(np.interp(t, an_s, yt_s))
            tips_curva.append(f"{t:.1f} años|Nominal {nom:.2f} %|"
                              f"Real {re_:.2f} %|Breakeven {nom - re_:.2f} %")
        else:
            tips_curva.append(f"{t:.2f} años|Nominal {nom:.2f} %")
    serie_curva = dict(W=980, padL=int(mL), padR=int(mR), n=n_cx,
                       tips=tips_curva)

    # ---- sobretasas de BONDES F por plazo --------------------------------
    lf = bn.sobretasas_bondesf(dv)
    st_buckets = []
    st_max = float(lf["st_pb"].max()) or 1.0
    for lo, hi, eti in [(0, 0.5, "0–6m"), (0.5, 1, "6m–1a"), (1, 2, "1–2a"),
                        (2, 3, "2–3a"), (3, 5, "3–5a"), (5, 9, "5a+")]:
        g = lf[(lf["anios"] >= lo) & (lf["anios"] < hi)]
        if not len(g):
            continue
        prom = float(g["st_pb"].mean())
        st_buckets.append(dict(
            eti=eti, stF=f"{prom:.1f} pb",
            w=f"{prom / st_max * 100:.0f}",
            tip=(f"BONDES F {eti}|Sobretasa promedio {prom:.1f} pb "
                 f"({len(g)} emisiones)"
                 f"|Rango {g['st_pb'].min():.0f}–{g['st_pb'].max():.0f} pb")))

    # ---- niveles por bono (datos puros del vector) -----------------------
    m_b = dv["bonos_m"]
    niveles = [dict(serie=str(se), venc=f"{pd.Timestamp(vc).date()}",
                    aniosF=f"{an:.2f}", ytmF=f"{yt:.2f} %",
                    durF=f"{du:.2f}", convF=f"{co:.1f}")
               for se, vc, an, yt, du, co in zip(
                   m_b["SERIE"], m_b["FECHA VCTO"], m_b["anios"],
                   m_b["ytm"], m_b["DURACION"], m_b["CONVEXIDAD"])]

    # ---- historico y percentiles -----------------------------------------
    hist = bn.historico_metricas()
    pcts = bn.percentiles_hoy(hist, a)
    n_hist = int(len(hist))
    hist_rows = []
    for x in pcts:
        if x["pct"] is not None:
            pct_f = f"p{x['pct']:.0f}"
            rango = f"{x['minimo']:.1f} – {x['maximo']:.1f}"
        else:
            pct_f = "—"
            rango = "—"
        hist_rows.append(dict(
            eti=x["etiqueta"],
            hoyF=(f"{x['hoy']:+.0f} {x['unidad']}" if x["unidad"] == "pb"
                  else f"{x['hoy']:.2f} {x['unidad']}"),
            pctF=pct_f, rangoF=rango))
    hist_nota = (
        f"{n_hist} sesión(es) acumulada(s) en data/hist_gob. Cada vector "
        f"nuevo agrega una foto; los percentiles se activan con 5 y ganan "
        f"sentido con 20+. Con historia suficiente, cada señal pasa de "
        f"“la curva está empinada” a “empinamiento en el "
        f"percentil X del año”.")

    # ================= ANALISIS =================
    r_max = float(res["residuo_pb"].abs().max()) or 1.0
    rich_cheap = [dict(
        name=r_.serie, valF=f"{r_.residuo_pb:+.0f} pb",
        wpos=f"{r_.residuo_pb / r_max * 100:.0f}" if r_.residuo_pb > 0 else 0,
        wneg=f"{abs(r_.residuo_pb) / r_max * 100:.0f}" if r_.residuo_pb < 0 else 0,
        color="var(--pos)" if r_.residuo_pb > 0 else "var(--neg)",
        tip=(f"BONO M {r_.serie}|Residuo {r_.residuo_pb:+.1f} pb vs curva "
             f"ajustada"))
        for r_ in res.itertuples()]

    choques = [-100, -50, 0, 50, 100]
    esc_filas = []
    for r_ in mx.itertuples():
        celdas = []
        for c in choques:
            tr = mx.loc[mx["serie"] == r_.serie, f"tr_{c:+d}"].iloc[0]
            dif = tr - cete1a
            intensidad = min(42, abs(dif) * 3.5)
            bg = (f"color-mix(in srgb, var(--pos) {intensidad:.0f}%, transparent)"
                  if dif >= 0 else
                  f"color-mix(in srgb, var(--neg) {intensidad:.0f}%, transparent)")
            celdas.append(dict(
                v=f"{tr:+.1f}", bg=bg,
                tip=(f"BONO M {r_.serie} · choque {c:+d} pb"
                     f"|Retorno total 12m: {tr:+.2f} %"
                     f"|vs CETE 1a ({cete1a:.2f} %): {dif:+.2f} pp")))
        esc_filas.append(dict(serie=r_.serie, aniosF=f"{r_.anios:.1f}",
                              celdas=celdas))
    escenarios = dict(
        filas=esc_filas,
        cols=[f"{c:+d} pb" if c else "Sin cambio" for c in choques],
        refF=f"{cete1a:.2f} %")

    mejor_total = cr.loc[cr["total_bp"].idxmax()]
    m_df = dv["bonos_m"].reset_index(drop=True)
    filas_carry = []
    for i, r_ in enumerate(cr.itertuples()):
        b = m_df.iloc[i]
        rpb = res_por_serie.get(r_.serie, 0.0)
        filas_carry.append(dict(
            serie=r_.serie, aniosF=f"{r_.anios:.2f}",
            ytmF=f"{r_.ytm:.2f} %", durF=f"{b['DURACION']:.2f}",
            resF=f"{rpb:+.0f}",
            resC=("var(--pos)" if rpb > 4 else
                  "var(--neg)" if rpb < -4 else "var(--ink3)"),
            carryF=f"{r_.carry_bp:+.0f}", rollF=f"{r_.roll_bp:+.0f}",
            totF=f"{r_.total_bp:+.0f}",
            totC=("var(--pos)" if r_.total_bp > 0 else "var(--neg)"),
            marca=("★ " if r_.serie == str(mejor_total["serie"]) else ""),
            tip=(f"BONO M {r_.serie}|Residuo {rpb:+.0f} pb · carry+roll 3m "
                 f"{r_.total_bp:+.0f} pb")))

    # ---- fijo contra flotante --------------------------------------------
    fvf = bn.fijo_vs_flotante(dv)
    fvf_rows = [dict(
        plazoF=f"{r_.anios:.1f}a", fijoF=f"{r_.fijo:.2f} %",
        stF=f"{r_.sobretasa_pb:.0f} pb",
        beF=f"{r_.breakeven_fondeo:.2f} %",
        alzaF=f"{r_.alza_requerida_pb:+.0f} pb",
        alzaC=("var(--pos)" if r_.alza_requerida_pb > 75 else
               "var(--warn)" if r_.alza_requerida_pb > 30 else "var(--neg)"),
        tip=(f"Plazo {r_.anios:.1f}a|Fijo (M) {r_.fijo:.2f} % vs BONDES F "
             f"fondeo+{r_.sobretasa_pb:.0f} pb|El flotante gana si el "
             f"fondeo promedia más de {r_.breakeven_fondeo:.2f} % "
             f"({r_.alza_requerida_pb:+.0f} pb vs hoy)"))
        for r_ in fvf.itertuples()]

    # ---- estrategias: respaldo + analisis autorado -----------------------
    n2 = a["nodo_m2"]
    estrategias = [
        dict(icon="▲", color="var(--pos)",
             titulo=f"Extensión corta: CETE 1a → BONO M {n2['SERIE']}",
             detalle=f"Pickup de {a['ext_corta']:+.0f} pb con duración "
                     f"{n2['DURACION']:.1f}."),
        dict(icon="★", color="var(--pos)",
             titulo=f"Carry+rolldown: {mejor_total['serie']}",
             detalle=f"{mejor_total['total_bp']:+.0f} pb a 3 meses."),
    ]
    autoria = ""
    ruta_analisis = DATOS / "analisis_deuda.json"
    if ruta_analisis.exists():
        import json
        try:
            aut = json.loads(ruta_analisis.read_text(encoding="utf-8"))
        except Exception:
            aut = None
        if aut and str(aut.get("fecha_vector")) == dv["fecha"].isoformat():
            tesis = [dict(icon="◎", color="var(--brand-lite)",
                          titulo=f"Tesis del día — {aut.get('autor', 'mesa')}",
                          detalle=str(aut.get("tesis", "")))]
            estrategias = tesis + [
                dict(icon=e.get("icon", "•"),
                     color=e.get("color", "var(--ink2)"),
                     titulo=str(e.get("titulo", "")),
                     detalle=str(e.get("detalle", "")))
                for e in aut.get("estrategias", [])]
            autoria = (f"Estrategias del análisis de mesa del "
                       f"{aut.get('fecha_analisis', '')} "
                       f"({aut.get('autor', '')}) para este vector. ")
        else:
            estrategias.insert(0, dict(
                icon="✎", color="var(--warn)",
                titulo="Análisis de mesa pendiente para este vector",
                detalle=("Tarjetas del motor automático. Corre la sesión "
                         "de análisis con Claude Code sobre el vector "
                         "vigente para publicar el análisis del día.")))

    rezago = (date.today() - dv["fecha"]).days
    if rezago >= 1:
        estrategias.insert(0, dict(
            icon="⚠", color="var(--warn)",
            titulo=(f"Vector del {dv['fecha'].day} "
                    f"{MES_UI[dv['fecha'].month]} — verifica vigencia"),
            detalle=(f"Cifras de hace {rezago} día(s). Sube el vector de "
                     f"hoy con el cargador de la parte superior.")))

    # ---- udibonos en escala nominal (comparativo contra Bonos M) ---------
    ESCENARIOS_PI = (3.0, 3.5, 4.0)
    un = bn.udibonos_nominalizados(dv, ESCENARIOS_PI)

    W2, H2, mL2, mR2, mT2, mB2 = 980.0, 300.0, 50.0, 66.0, 16.0, 30.0
    u_lo = float(np.floor(un["nom_3.0"].min())) - 0.4
    u_hi = float(np.ceil(dv["bonos_m"]["ytm"].max())) + 0.3

    def X2(t):
        return mL2 + (max(t, 0.0) / t_max) ** 0.5 * (W2 - mL2 - mR2)

    def Y2(y):
        return mT2 + (u_hi - y) / (u_hi - u_lo) * (H2 - mT2 - mB2)

    linea_m2 = " ".join(f"{X2(t):.1f},{Y2(y):.1f}" for t, y in
                        zip(dv["bonos_m"]["anios"], dv["bonos_m"]["ytm"]))
    lineas_pi = {}
    for pi in ESCENARIOS_PI:
        lineas_pi[pi] = " ".join(
            f"{X2(t):.1f},{Y2(y):.1f}" for t, y in
            zip(un["anios"], un[f"nom_{pi:.1f}"]))

    u_gmay = [dict(y=f"{Y2(v):.1f}", label=f"{v:.0f}%")
              for v in np.arange(np.ceil(u_lo), u_hi, 1.0)]
    u_ticks = [dict(x=f"{X2(t):.1f}", label=lab)
               for t, lab in [(1, "1a"), (2, "2a"), (5, "5a"), (10, "10a"),
                              (20, "20a"), (28.5, "30a")] if t <= t_max]
    fin_x = f"{X2(float(un['anios'].max())):.1f}"
    etiquetas_pi = [dict(x=fin_x, y=f"{Y2(float(un[f'nom_{pi:.1f}'].iloc[-1])):.1f}",
                         label=f"π {pi:.1f} %")
                    for pi in ESCENARIOS_PI]
    fin_m_y = f"{Y2(float(dv['bonos_m']['ytm'].iloc[-1])):.1f}"

    udinom = dict(lineaM=linea_m2,
                  l30=lineas_pi[3.0], l35=lineas_pi[3.5], l40=lineas_pi[4.0],
                  gMay=u_gmay, ticksX=u_ticks,
                  etiquetasPi=etiquetas_pi, finX=f"{X2(t_max):.1f}",
                  finMy=fin_m_y)

    # crosshair del comparativo
    an_u = un["anios"].values
    n_u = 80
    tips_u = []
    for i in range(n_u):
        t = ((i / (n_u - 1)) ** 2) * t_max
        m_ = float(np.interp(t, an_, yt_))
        if t >= float(an_u.min()):
            partes = [f"{t:.1f} años", f"Bono M {m_:.2f} %"]
            for pi in ESCENARIOS_PI:
                nv = float(np.interp(t, an_u, un[f"nom_{pi:.1f}"].values))
                partes.append(f"UDI π{pi:.1f}%: {nv:.2f} % "
                              f"({(nv - m_) * 100:+.0f} pb)")
            be_ = float(np.interp(t, an_u, un["be_nodo"].values))
            partes.append(f"π de indiferencia: {be_:.2f} %")
            tips_u.append("|".join(partes))
        else:
            tips_u.append(f"{t:.2f} años|Bono M {m_:.2f} %")
    serie_udinom_hover = dict(W=980, padL=int(mL2), padR=int(mR2),
                              n=n_u, tips=tips_u)

    # tabla de decision por nodo
    udinom_filas = []
    for r_ in un.itertuples():
        v30 = getattr(r_, "vent_3.0", None)
        celdas_v = {}
        for pi in ESCENARIOS_PI:
            v = float(un.loc[un["serie"] == r_.serie,
                             f"vent_{pi:.1f}"].iloc[0])
            clave = f"v{str(pi).replace('.', '')}"
            celdas_v[clave + "F"] = f"{v:+.0f}"
            celdas_v[clave + "C"] = ("var(--pos)" if v > 0 else "var(--neg)")
        udinom_filas.append(dict(
            serie=r_.serie, aniosF=f"{r_.anios:.1f}",
            realF=f"{r_.real:.2f} %", mF=f"{r_.m_interp:.2f} %",
            beF=f"{r_.be_nodo:.2f} %",
            tip=(f"UDIBONO {r_.serie}|Real {r_.real:.2f} % · M mismo plazo "
                 f"{r_.m_interp:.2f} %|Gana al nominal sólo si la inflación "
                 f"promedio de {r_.anios:.0f} años supera {r_.be_nodo:.2f} %"),
            **celdas_v))

    fwd_txt = " · ".join(f"{f['t1']}a→{f['t2']}a {f['fwd']:.2f} %"
                         for f in fwds)

    # ================= VALUACION =================
    val_rows = [
        dict(clase="CETES", n=f"{len(dv['cetes'])}",
             met="Descuento anualizado 360",
             difF=f"{dv['validacion']['cetes_pb']:.3f} pb"),
        dict(clase="Bonos M", n=f"{len(dv['bonos_m'])}",
             met="Cupón semestral 182/360, bisección",
             difF=f"{dv['validacion']['m_pb']:.3f} pb"),
        dict(clase="Udibonos", n=f"{len(dv['udibonos'])}",
             met="182/360 en UDIs (UDI implícita)",
             difF=f"{dv['validacion']['udibonos_pb']:.3f} pb"),
        dict(clase="Bondes F", n=f"{len(dv['bondesf'])}",
             met="Tasa oficial del vector (flotante)", difF="—"),
    ]
    convenciones = (
        "La fuente primaria de tasas es la columna TASA DE RENDIMIENTO del "
        "vector. En cada carga, un solver propio revalúa cada instrumento "
        "desde su precio y compara contra la oficial (tabla izquierda): si "
        "un vector llegara con precios y tasas inconsistentes, la "
        "desviación lo delata. La UDI se recupera del interés acumulado en "
        "pesos de cada udibono (mediana entre emisiones). Duración "
        "modificada del vector (base 360, verificada contra la Macaulay en "
        "días). Carry fondeado al CETE de 91d y rolldown con curva sin "
        "cambios. Análisis de mesa; no constituye una recomendación de "
        "inversión.")

    cb = bn.credito_buckets()
    cred_rows, cred_pick, cred_nota = [], "", ""
    if cb is not None:
        for f in cb["tabla"]:
            fila = dict(rating=f["rating"], nF=f"{f['n']}")
            for pz in ["0-1a", "1-3a", "3-5a", "5a+"]:
                v = f.get(pz)
                clave = {"0-1a": "p01", "1-3a": "p13",
                         "3-5a": "p35", "5a+": "p5m"}[pz]
                fila[clave] = f"{v[0]:.0f} ({v[1]})" if v else "—"
            cred_rows.append(fila)
        cred_pick = " · ".join(f"{k}: {v:+.0f} pb" for k, v in cb["pickups"])
        cred_nota = (
            f"{cb['total']:,} títulos corporativos de tasa revisable del "
            f"vector; mediana de sobretasa en pb (n) por calificación y "
            f"plazo. Se usa mediana porque el universo mezcla papel "
            f"quirografario, bancario y estructurado — la curaduría por "
            f"sector está pendiente y explica anomalías como el escalón "
            f"AA→A. {cb['sin_calif']} títulos sin calificación excluidos "
            f"del cuadro. Uso: cualquier papel que ofrezca el mercado se "
            f"compara contra la mediana de su celda.")

    return dict(
        fechaF=f"{dv['fecha'].day} {MES_UI[dv['fecha'].month]} {dv['fecha'].year}",
        udiF=f"{dv['udi']:.4f}",
        kpis=[
            dict(label="CETE 28d", value=f"{a['cete28']:.2f} %"),
            dict(label="CETE 1a", value=f"{a['cete364']:.2f} %"),
            dict(label="BONO M 2a", value=f"{a['m2']:.2f} %"),
            dict(label="BONO M 10a", value=f"{a['m10']:.2f} %"),
            dict(label="2s10s", value=f"{a['p2s10']:+.0f} pb"),
            dict(label="Fly 2s5s10s", value=f"{a['fly_2_5_10']:+.0f} pb"),
            dict(label="BE 10a", value=f"{a['be10']:.2f} %"),
            dict(label="Fondeo (91d)", value=f"{fondeo:.2f} %"),
        ],
        curva=curva, serieCurva=serie_curva, udinom=udinom,
        serieUdinom=serie_udinom_hover, udinomFilas=udinom_filas,
        stBuckets=st_buckets, niveles=niveles,
        histRows=hist_rows, histNota=hist_nota, histN=f"{n_hist}",
        richCheap=rich_cheap, escenarios=escenarios,
        filasCarry=filas_carry, fvfRows=fvf_rows,
        estrategias=estrategias, fwdTxt=f"Forwards: {fwd_txt}",
        valRows=val_rows, convenciones=convenciones,
        credRows=cred_rows, credPick=cred_pick, credNota=cred_nota,
        notaCurva=("Nodos M coloreados por valor relativo (verde barato, "
                   "rojo caro) y etiquetados por serie; CETES en ámbar, "
                   "Udibonos (tasa real) en verde, fondeo punteado. Recorre "
                   "la curva con el cursor para nominal, real y breakeven."),
        notaSt=("La sobretasa de BONDES F tiene curva propia: extender el "
                "plazo del flotante multiplica el spread sin tomar duración "
                "de tasa — la extensión más barata que existe para la "
                "caja."),
        notaRichCheap=("Residuo de cada Bono M contra la curva ajustada "
                       "(cúbica en raíz del plazo). Positivo rinde de más "
                       "(barato); negativo, de menos (caro)."),
        notaEscenarios=("Retorno total a 12 meses: carry + rolldown + "
                        "precio (−dur·Δy + ½·conv·Δy²) bajo choques "
                        "paralelos. Verde gana al CETE de 1a; la fila donde "
                        "el verde resiste +100 pb es la extensión "
                        "defendible."),
        notaUdinom=("Curva de Udibonos transformada a tasa nominal "
                    "equivalente con Fisher exacto bajo tres escenarios de "
                    "inflación promedio (π). Donde la línea del escenario "
                    "queda ARRIBA de la curva M, el udibono gana al nominal "
                    "en ese plazo bajo ese escenario; abajo, pierde. La π "
                    "de indiferencia de cada nodo es su breakeven: el "
                    "escenario que iguala ambas curvas."),
        notaFvf=("Fijo contra flotante por plazo: pb de alza PROMEDIO del "
                 "fondeo que hacen falta para que el BONDES F gane al Bono "
                 "M. Verde: el fijo tiene colchón amplio; rojo: el "
                 "flotante casi empata."),
        notaAutoria=(autoria or "Motor automático de reglas (sin análisis "
                     "autorado para este vector)."),
    )


def _seccion_deuda(html: str, deu: dict) -> str:
    # ---- navegacion agrupada, titulos y banderas -------------------------
    html = _sub(html,
        "const NAV = [['resumen','Resumen','01'],",
        "const NAV = [['#','Renta variable'],['resumen','Resumen','01'],")
    html = _sub(html,
        "['rotacion','Rotación','09'],['configuracion','Configuración','10']",
        "['rotacion','Rotación','09'],"
        "['#','Renta fija'],"
        "['deuda-mdo','Mercado · Vector','10'],"
        "['deuda-ana','Análisis · Estrategias','11'],"
        "['deuda-val','Valuación · Crédito','12'],"
        "['#','General'],['configuracion','Configuración','13']")
    html = _sub(html,
        "const nav = NAV.map(([id, label, code]) => ({ label, code, "
        "go: () => this.setState({ tab: id }), "
        "stl: this.navStl(this.state.tab === id) }));",
        "const nav = NAV.map(([id, label, code]) => (id === '#' "
        "? { label, code: '', go: () => {}, stl: 'display:block;width:100%;"
        "text-align:left;font:700 8.5px/1 var(--sans);letter-spacing:.14em;"
        "text-transform:uppercase;color:var(--ink3);padding:14px 11px 5px;"
        "border:none;border-top:1px solid var(--border);margin-top:6px;"
        "background:transparent;cursor:default;pointer-events:none' } "
        ": { label, code, go: () => this.setState({ tab: id }), "
        "stl: this.navStl(this.state.tab === id) }));")
    html = _sub(html,
        "configuracion: ['Configuración', 'Fuentes de datos",
        "'deuda-mdo': ['Renta fija · Mercado', 'Vector Valmer del "
        + deu["fechaF"] + " · curvas, sobretasas, niveles e histórico'], "
        "'deuda-ana': ['Renta fija · Análisis y estrategias', 'Tesis de la "
        "mesa, valor relativo, escenarios y carry'], "
        "'deuda-val': ['Renta fija · Valuación y crédito', 'Verificación "
        "de precios, convenciones y sobretasas corporativas'], "
        "configuracion: ['Configuración', 'Fuentes de datos")
    html = _sub(html,
        "isRotacion: this.state.tab === 'rotacion', rot: ",
        "isDeudaMdo: this.state.tab === 'deuda-mdo', "
        "isDeudaAna: this.state.tab === 'deuda-ana', "
        "isDeudaVal: this.state.tab === 'deuda-val', "
        "deu: " + js(deu)
        + ", isRotacion: this.state.tab === 'rotacion', rot: ")

    TD = '<\\u002Fdiv>'
    card = ('background:var(--surf);border:1px solid var(--border);'
            'border-radius:6px;padding:14px 16px')
    header = ('font:700 10px/1 var(--sans);letter-spacing:.14em;'
              'text-transform:uppercase;color:var(--ink3);margin-bottom:12px')
    nota = ('font:400 10.5px/1.55 var(--sans);color:var(--ink3);'
            'margin-top:10px;border-top:1px dashed var(--border);'
            'padding-top:8px')
    th = ('text-align:right;font:700 9px/1 var(--sans);'
          'letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);'
          'padding:8px 9px')
    th_izq = th.replace('text-align:right', 'text-align:left')
    td = 'padding:6px 9px;text-align:right;font-family:var(--mono)'

    def chip(color, texto):
        return ('<span style=\\"display:inline-flex;align-items:center;'
                'gap:5px\\"><span style=\\"width:9px;height:9px;'
                'border-radius:2px;background:' + color
                + ';display:inline-block\\"><\\u002Fspan>' + texto
                + '<\\u002Fspan>')

    def kpis_strip():
        return ('<div style=\\"display:grid;'
                'grid-template-columns:repeat(8,1fr);gap:10px;'
                'margin-bottom:14px\\">'
                '<sc-for list=\\"{{ deu.kpis }}\\" as=\\"k\\" '
                'hint-placeholder-count=\\"8\\">'
                '<div style=\\"' + card.replace('padding:14px 16px',
                                                'padding:10px 12px')
                + '\\"><div style=\\"font:600 8.5px/1 var(--sans);'
                'letter-spacing:.1em;text-transform:uppercase;'
                'color:var(--ink3)\\">{{ k.label }}' + TD +
                '<div style=\\"font:600 15px/1.3 var(--mono);'
                'color:var(--ink);margin-top:4px\\">{{ k.value }}' + TD + TD
                + '<\\u002Fsc-for>' + TD)

    def tarjeta_estrategia():
        return ('<sc-for list=\\"{{ deu.estrategias }}\\" as=\\"g\\" '
                'hint-placeholder-count=\\"6\\">'
                '<div style=\\"display:flex;gap:10px;padding:9px 10px;'
                'margin-bottom:7px;background:var(--surf2);'
                'border-radius:5px;border-left:3px solid {{ g.color }}\\">'
                '<div style=\\"color:{{ g.color }};font-size:12px\\">'
                '{{ g.icon }}' + TD +
                '<div><div style=\\"font:600 11.5px/1.35 var(--sans);'
                'color:var(--ink)\\">{{ g.titulo }}' + TD +
                '<div style=\\"font:400 10.5px/1.5 var(--sans);'
                'color:var(--ink2);margin-top:2px\\">{{ g.detalle }}'
                + TD + TD + TD + '<\\u002Fsc-for>')

    # ============ SECCION 1: MERCADO ============
    sec_mercado = (
        '<!-- ================= RF MERCADO ================= -->\\n'
        '      <sc-if value=\\"{{ isDeudaMdo }}\\" '
        'hint-placeholder-val=\\"{{ false }}\\">\\n      '
        + kpis_strip() + '\\n'
        '      <div style=\\"' + card + ';margin-bottom:14px\\">'
        '<div style=\\"' + header + '\\">Curvas de rendimiento · UDI '
        '{{ deu.udiF }}<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · {{ deu.fwdTxt }}<\\u002Fspan>' + TD +
        '<svg viewBox=\\"0 0 980 380\\" style=\\"width:100%;height:auto\\">'
        '<sc-for list=\\"{{ deu.curva.gMen }}\\" as=\\"g\\" '
        'hint-placeholder-count=\\"7\\">'
        '<line x1=\\"50\\" y1=\\"{{ g.y }}\\" x2=\\"962\\" '
        'y2=\\"{{ g.y }}\\" stroke=\\"var(--border)\\" opacity=\\".45\\" '
        'stroke-dasharray=\\"2 5\\"><\\u002Fline><\\u002Fsc-for>'
        '<sc-for list=\\"{{ deu.curva.gMay }}\\" as=\\"g\\" '
        'hint-placeholder-count=\\"8\\">'
        '<line x1=\\"50\\" y1=\\"{{ g.y }}\\" x2=\\"962\\" '
        'y2=\\"{{ g.y }}\\" stroke=\\"var(--border)\\" '
        'stroke-dasharray=\\"2 4\\"><\\u002Fline>'
        '<text x=\\"8\\" y=\\"{{ g.y }}\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"9.5\\" fill=\\"var(--ink3)\\" dy=\\"3\\">'
        '{{ g.label }}<\\u002Ftext><\\u002Fsc-for>'
        '<sc-for list=\\"{{ deu.curva.ticksX }}\\" as=\\"tx\\" '
        'hint-placeholder-count=\\"12\\">'
        '<line x1=\\"{{ tx.x }}\\" y1=\\"346\\" x2=\\"{{ tx.x }}\\" '
        'y2=\\"352\\" stroke=\\"var(--border2)\\"><\\u002Fline>'
        '<text x=\\"{{ tx.x }}\\" y=\\"366\\" text-anchor=\\"middle\\" '
        'font-family=\\"var(--mono)\\" font-size=\\"9.5\\" '
        'fill=\\"var(--ink3)\\">{{ tx.label }}<\\u002Ftext><\\u002Fsc-for>'
        '<line x1=\\"50\\" y1=\\"{{ deu.curva.fondeoY }}\\" x2=\\"962\\" '
        'y2=\\"{{ deu.curva.fondeoY }}\\" stroke=\\"var(--ink3)\\" '
        'stroke-width=\\"1.2\\" stroke-dasharray=\\"7 4\\" '
        'opacity=\\".8\\"><\\u002Fline>'
        '<text x=\\"958\\" y=\\"{{ deu.curva.fondeoY }}\\" '
        'text-anchor=\\"end\\" dy=\\"-5\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"9\\" fill=\\"var(--ink3)\\">'
        '{{ deu.curva.fondeoF }}<\\u002Ftext>'
        '<polyline points=\\"{{ deu.curva.lineaC }}\\" fill=\\"none\\" '
        'stroke=\\"var(--warn)\\" stroke-width=\\"2\\" '
        'opacity=\\".9\\"><\\u002Fpolyline>'
        '<polyline data-chart=\\"curva\\" '
        'points=\\"{{ deu.curva.lineaM }}\\" fill=\\"none\\" '
        'stroke=\\"var(--brand-lite)\\" stroke-width=\\"2.6\\">'
        '<\\u002Fpolyline>'
        '<polyline points=\\"{{ deu.curva.lineaS }}\\" fill=\\"none\\" '
        'stroke=\\"var(--pos)\\" stroke-width=\\"2\\" opacity=\\".9\\">'
        '<\\u002Fpolyline>'
        '<sc-for list=\\"{{ deu.curva.etiquetasM }}\\" as=\\"e\\" '
        'hint-placeholder-count=\\"18\\">'
        '<text x=\\"{{ e.x }}\\" y=\\"{{ e.y }}\\" '
        'text-anchor=\\"middle\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"8\\" fill=\\"var(--ink3)\\">{{ e.label }}'
        '<\\u002Ftext><\\u002Fsc-for>'
        '<sc-for list=\\"{{ deu.curva.puntos }}\\" as=\\"p\\" '
        'hint-placeholder-count=\\"40\\">'
        '<circle data-tip=\\"{{ p.tip }}\\" cx=\\"{{ p.cx }}\\" '
        'cy=\\"{{ p.cy }}\\" r=\\"{{ p.r }}\\" fill=\\"{{ p.color }}\\" '
        'stroke=\\"var(--surf)\\" stroke-width=\\"1.4\\"><\\u002Fcircle>'
        '<\\u002Fsc-for><\\u002Fsvg>'
        '<div style=\\"display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;'
        'font-family:var(--mono);font-size:10px\\">'
        + chip('var(--brand-lite)', 'Bonos M (nominal)')
        + chip('var(--warn)', 'CETES')
        + chip('var(--pos)', 'Udibonos (real) / nodo barato')
        + chip('var(--neg)', 'nodo caro') + TD +
        '<div style=\\"' + nota + '\\">{{ deu.notaCurva }}' + TD + TD + '\\n'
        # sobretasas | historico
        '      <div style=\\"display:grid;grid-template-columns:1fr 1fr;'
        'gap:14px;margin-bottom:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Sobretasas BONDES F por plazo'
        + TD +
        '<sc-for list=\\"{{ deu.stBuckets }}\\" as=\\"b\\" '
        'hint-placeholder-count=\\"6\\">'
        '<div data-tip=\\"{{ b.tip }}\\" style=\\"display:flex;'
        'align-items:center;gap:10px;margin-bottom:9.5px\\">'
        '<div style=\\"width:52px;font-family:var(--mono);font-size:10.5px;'
        'color:var(--ink2)\\">{{ b.eti }}' + TD +
        '<div style=\\"flex:1;height:13px;background:var(--surf3);'
        'border-radius:3px;overflow:hidden\\">'
        '<div style=\\"height:100%;width:{{ b.w }}%;'
        'background:var(--brand-lite);border-radius:3px\\">' + TD + TD +
        '<div style=\\"width:56px;text-align:right;'
        'font-family:var(--mono);font-size:10.5px;color:var(--ink)\\">'
        '{{ b.stF }}' + TD + TD + '<\\u002Fsc-for>'
        '<div style=\\"' + nota + '\\">{{ deu.notaSt }}' + TD + TD + '\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Histórico de la curva '
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\">· {{ deu.histN }} sesión(es)<\\u002Fspan>'
        + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Métrica<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Hoy<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Percentil<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Rango hist.<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.histRows }}\\" as=\\"h\\" '
        'hint-placeholder-count=\\"6\\">'
        '<sc-raw-tr style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font-size:10.5px;'
        'color:var(--ink2)\\">{{ h.eti }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ h.hoyF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ h.pctF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink3)\\">'
        '{{ h.rangoF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>'
        '<div style=\\"' + nota + '\\">{{ deu.histNota }}' + TD + TD + TD
        + '\\n'
        # niveles
        '      <div style=\\"' + card.replace('padding:14px 16px',
                                              'padding:0;overflow:hidden')
        + '\\">'
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Bono M<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Vencimiento'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Años<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">YTM<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Duración<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Convexidad<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.niveles }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"18\\">'
        '<sc-raw-tr style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ f.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"padding:6px 9px;font-family:var(--mono);'
        'color:var(--ink2)\\">{{ f.venc }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.aniosF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ f.ytmF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.durF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.convF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        '      <\\u002Fsc-if>\\n\\n      ')

    # ============ SECCION 2: ANALISIS ============
    sec_analisis = (
        '<!-- ================= RF ANALISIS ================= -->\\n'
        '      <sc-if value=\\"{{ isDeudaAna }}\\" '
        'hint-placeholder-val=\\"{{ false }}\\">\\n'
        '      <div style=\\"display:grid;grid-template-columns:1.3fr 1fr;'
        'gap:14px;margin-bottom:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Estrategias de la mesa' + TD
        + tarjeta_estrategia() +
        '<div style=\\"' + nota + '\\">{{ deu.notaAutoria }}' + TD + TD
        + '\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Valor relativo por bono'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · residuo vs curva<\\u002Fspan>' + TD +
        '<sc-for list=\\"{{ deu.richCheap }}\\" as=\\"b\\" '
        'hint-placeholder-count=\\"18\\">'
        '<div data-tip=\\"{{ b.tip }}\\" style=\\"display:flex;'
        'align-items:center;gap:8px;margin-bottom:4px\\">'
        '<div style=\\"width:56px;font-family:var(--mono);font-size:10px;'
        'color:var(--ink2)\\">{{ b.name }}' + TD +
        '<div style=\\"flex:1;display:flex;align-items:center\\">'
        '<div style=\\"width:50%;display:flex;justify-content:flex-end\\">'
        '<div style=\\"height:9px;width:{{ b.wneg }}%;'
        'background:var(--neg);border-radius:2px 0 0 2px\\">' + TD + TD +
        '<div style=\\"width:1px;height:13px;background:var(--border2)\\">'
        + TD + '<div style=\\"width:50%\\">'
        '<div style=\\"height:9px;width:{{ b.wpos }}%;'
        'background:var(--pos);border-radius:0 2px 2px 0\\">' + TD + TD + TD
        + '<div style=\\"width:52px;text-align:right;'
        'font-family:var(--mono);font-size:10px;color:{{ b.color }}\\">'
        '{{ b.valF }}' + TD + TD + '<\\u002Fsc-for>'
        '<div style=\\"' + nota + '\\">{{ deu.notaRichCheap }}' + TD + TD +
        TD + '\\n'
        # matriz
        '      <div style=\\"' + card + ';margin-bottom:14px\\">'
        '<div style=\\"' + header + '\\">Retorno total a 12 meses por '
        'escenario<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · verde gana al CETE 1a '
        '({{ deu.escenarios.refF }})<\\u002Fspan>' + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Bono M<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Años<\\u002Fsc-raw-th>'
        '<sc-for list=\\"{{ deu.escenarios.cols }}\\" as=\\"c\\" '
        'hint-placeholder-count=\\"5\\">'
        '<sc-raw-th style=\\"' + th + ';text-align:center\\">{{ c }}'
        '<\\u002Fsc-raw-th><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.escenarios.filas }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"18\\">'
        '<sc-raw-tr style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:5px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ f.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink3)\\">'
        '{{ f.aniosF }}<\\u002Fsc-raw-td>'
        '<sc-for list=\\"{{ f.celdas }}\\" as=\\"c\\" '
        'hint-placeholder-count=\\"5\\">'
        '<sc-raw-td data-tip=\\"{{ c.tip }}\\" style=\\"padding:5px 9px;'
        'text-align:center;font-family:var(--mono);font-weight:600;'
        'color:var(--ink);background:{{ c.bg }}\\">{{ c.v }}'
        '<\\u002Fsc-raw-td><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>'
        '<div style=\\"' + nota + '\\">{{ deu.notaEscenarios }}' + TD + TD
        + '\\n'
        # udibonos nominalizados vs bonos M
        '      <div style=\\"' + card + ';margin-bottom:14px\\">'
        '<div style=\\"' + header + '\\">Udibonos en escala nominal vs '
        'Bonos M<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · Fisher exacto por escenario de inflación '
        'promedio<\\u002Fspan>' + TD +
        '<svg viewBox=\\"0 0 980 300\\" style=\\"width:100%;height:auto\\">'
        '<sc-for list=\\"{{ deu.udinom.gMay }}\\" as=\\"g\\" '
        'hint-placeholder-count=\\"5\\">'
        '<line x1=\\"50\\" y1=\\"{{ g.y }}\\" x2=\\"914\\" '
        'y2=\\"{{ g.y }}\\" stroke=\\"var(--border)\\" '
        'stroke-dasharray=\\"2 4\\"><\\u002Fline>'
        '<text x=\\"8\\" y=\\"{{ g.y }}\\" font-family=\\"var(--mono)\\" '
        'font-size=\\"9.5\\" fill=\\"var(--ink3)\\" dy=\\"3\\">'
        '{{ g.label }}<\\u002Ftext><\\u002Fsc-for>'
        '<sc-for list=\\"{{ deu.udinom.ticksX }}\\" as=\\"tx\\" '
        'hint-placeholder-count=\\"6\\">'
        '<text x=\\"{{ tx.x }}\\" y=\\"292\\" text-anchor=\\"middle\\" '
        'font-family=\\"var(--mono)\\" font-size=\\"9.5\\" '
        'fill=\\"var(--ink3)\\">{{ tx.label }}<\\u002Ftext><\\u002Fsc-for>'
        '<polyline data-chart=\\"udinom\\" '
        'points=\\"{{ deu.udinom.lineaM }}\\" fill=\\"none\\" '
        'stroke=\\"var(--brand-lite)\\" stroke-width=\\"2.6\\">'
        '<\\u002Fpolyline>'
        '<polyline points=\\"{{ deu.udinom.l30 }}\\" fill=\\"none\\" '
        'stroke=\\"var(--pos)\\" stroke-width=\\"1.8\\" opacity=\\".65\\" '
        'stroke-dasharray=\\"3 5\\"><\\u002Fpolyline>'
        '<polyline points=\\"{{ deu.udinom.l35 }}\\" fill=\\"none\\" '
        'stroke=\\"var(--pos)\\" stroke-width=\\"2\\" opacity=\\".85\\" '
        'stroke-dasharray=\\"9 5\\"><\\u002Fpolyline>'
        '<polyline points=\\"{{ deu.udinom.l40 }}\\" fill=\\"none\\" '
        'stroke=\\"var(--pos)\\" stroke-width=\\"2.4\\"><\\u002Fpolyline>'
        '<text x=\\"{{ deu.udinom.finX }}\\" '
        'y=\\"{{ deu.udinom.finMy }}\\" dy=\\"4\\" '
        'font-family=\\"var(--mono)\\" font-size=\\"9\\" '
        'font-weight=\\"700\\" fill=\\"var(--brand-lite)\\">Bonos M'
        '<\\u002Ftext>'
        '<sc-for list=\\"{{ deu.udinom.etiquetasPi }}\\" as=\\"e\\" '
        'hint-placeholder-count=\\"3\\">'
        '<text x=\\"{{ e.x }}\\" y=\\"{{ e.y }}\\" dx=\\"6\\" dy=\\"3\\" '
        'font-family=\\"var(--mono)\\" font-size=\\"9\\" '
        'font-weight=\\"700\\" fill=\\"var(--pos)\\">{{ e.label }}'
        '<\\u002Ftext><\\u002Fsc-for><\\u002Fsvg>'
        '<div style=\\"' + nota + '\\">{{ deu.notaUdinom }}' + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px;margin-top:10px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Udibono<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Años<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Real<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">M mismo plazo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">π indiferencia'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">vs M @3.0% (pb)'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">@3.5%<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">@4.0%<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.udinomFilas }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"14\\">'
        '<sc-raw-tr data-tip=\\"{{ f.tip }}\\" '
        'style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ f.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.aniosF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ f.realF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">{{ f.mF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ f.beF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:{{ f.v30C }}\\">{{ f.v30F }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:{{ f.v35C }}\\">{{ f.v35F }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:{{ f.v40C }}\\">{{ f.v40F }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        # fijo vs flotante | carry
        '      <div style=\\"display:grid;grid-template-columns:1fr 1.5fr;'
        'gap:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Fijo vs flotante'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · breakeven de fondeo<\\u002Fspan>' + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Plazo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Fijo (M)<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Sobretasa<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Fondeo BE<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Alza req.<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.fvfRows }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"5\\">'
        '<sc-raw-tr data-tip=\\"{{ f.tip }}\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font-family:var(--mono);'
        'color:var(--ink)\\">{{ f.plazoF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ f.fijoF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.stF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.beF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:700;'
        'color:{{ f.alzaC }}\\">{{ f.alzaF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>'
        '<div style=\\"' + nota + '\\">{{ deu.notaFvf }}' + TD + TD + '\\n'
        '        <div style=\\"' + card.replace('padding:14px 16px',
                                                'padding:0;overflow:hidden')
        + '\\">'
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Bono M<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Años<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">YTM<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Dur.<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Residuo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Carry 3m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Roll 3m<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Total pb<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.filasCarry }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"18\\">'
        '<sc-raw-tr data-tip=\\"{{ f.tip }}\\" '
        'style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ f.marca }}{{ f.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.aniosF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ f.ytmF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.durF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:{{ f.resC }}\\">{{ f.resF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.carryF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.rollF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:700;'
        'color:{{ f.totC }}\\">{{ f.totF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + TD + '\\n'
        '      <\\u002Fsc-if>\\n\\n      ')

    # ============ SECCION 3: VALUACION ============
    sec_valuacion = (
        '<!-- ================= RF VALUACION ================= -->\\n'
        '      <sc-if value=\\"{{ isDeudaVal }}\\" '
        'hint-placeholder-val=\\"{{ false }}\\">\\n'
        '      <div style=\\"display:grid;grid-template-columns:1fr 1fr;'
        'gap:14px;margin-bottom:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Verificación de valuación'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · vector del {{ deu.fechaF }} · UDI '
        '{{ deu.udiF }}<\\u002Fspan>' + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Clase<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Títulos<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Convención'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Desv. máx.<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.valRows }}\\" as=\\"v\\" '
        'hint-placeholder-count=\\"4\\">'
        '<sc-raw-tr style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:7px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ v.clase }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">{{ v.n }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"padding:7px 9px;font-size:10px;'
        'color:var(--ink2)\\">{{ v.met }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--pos)\\">{{ v.difF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Convenciones y método' + TD +
        '<div style=\\"font:400 11px/1.65 var(--sans);'
        'color:var(--ink2)\\">{{ deu.convenciones }}' + TD + TD + TD + '\\n'
        # credito corporativo
        '      <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Crédito corporativo · sobretasa '
        'mediana por calificación y plazo'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · pickup {{ deu.credPick }}<\\u002Fspan>'
        + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Calificación'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Títulos<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">0–1a<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">1–3a<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">3–5a<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">5a+<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ deu.credRows }}\\" as=\\"c\\" '
        'hint-placeholder-count=\\"4\\">'
        '<sc-raw-tr style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:7px 9px;font:700 11px var(--mono);'
        'color:var(--ink)\\">{{ c.rating }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink3)\\">{{ c.nF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ c.p01 }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ c.p13 }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ c.p35 }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ c.p5m }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>'
        '<div style=\\"' + nota + '\\">{{ deu.credNota }}' + TD + TD + '\\n'
        '      <\\u002Fsc-if>\\n\\n      ')

    html = _sub(html,
        '<!-- ================= CONFIGURACIÓN ================= -->',
        sec_mercado + sec_analisis + sec_valuacion
        + '<!-- ================= CONFIGURACIÓN ================= -->')
    return html


def _verificar_bundle(html: str) -> None:
    """
    Los bloques __bundler/* del tablero son JSON: una inyeccion con escapes
    mal balanceados los rompe y el sintoma en el navegador es una pagina en
    blanco sin error. Mejor tronar aqui, con posicion y contexto.
    """
    import json
    for m in re.finditer(r'<script type="(__bundler/[a-z_]+)">', html):
        fin = html.find("</script>", m.end())
        contenido = html[m.end():fin].strip()
        if not contenido.startswith(("{", "[", '"')):
            continue
        try:
            json.loads(contenido)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Inyeccion invalida: el bloque {m.group(1)} dejo de ser "
                f"JSON en la posicion {e.pos}: "
                f"...{contenido[max(0, e.pos - 60):e.pos + 30]!r}...") from e


# --------------------------------------------------------------------------
# Cartera de renta fija y ALM: cuarta pestana del grupo de deuda
# --------------------------------------------------------------------------
# Posiciones propias de deuda valuadas contra el vector (DV01 monetario),
# posicionamiento KRD contra el benchmark gubernamental por circulacion,
# calce ALM real contra el pasivo y estructuras DV01-neutral listas para
# boleta. La cartera vive en data/posiciones_deuda.csv y el pasivo en
# data/pasivo_real.csv, ambos editables.

from . import deuda_port as dpo


def _cartera_rf() -> dict | None:
    dv = bn.cargar()
    if dv is None:
        return None
    cart = dpo.valuar_cartera(dv)
    if not len(cart):
        return None

    total = float(cart["valor"].sum())
    dv01_tot = float(cart["dv01"].sum())
    dur_pond = float((cart["DURACION"] * cart["valor"]).sum() / total)
    carry_anual = float((cart["ytm"] * cart["valor"]).sum() / total)
    pnl = float(cart["pnl"].sum())
    hist = bn.historico_metricas()
    te = dpo.te_ex_ante(hist)

    filas = []
    for r in cart.itertuples():
        ok = bool(r.encontrado)
        filas.append(dict(
            clase=str(r.tipo_valor), serie=str(r.serie),
            titF=f"{r.titulos:,.0f}",
            costoF=f"{r.costo_unit:,.3f}" if ok else "—",
            sucioF=f"{r.sucio:,.3f}" if ok else "sin precio",
            valorF=f"{r.valor:,.0f}" if ok else "—",
            ytmF=f"{r.ytm:.2f} %" if ok else "—",
            durF=f"{r.DURACION:.2f}" if ok else "—",
            dv01F=f"{r.dv01:,.0f}" if ok else "—",
            pesoF=f"{r.peso_pct:.1f} %" if ok else "—",
            pnlF=f"{r.pnl:+,.0f}" if ok else "—",
            pnlC=("var(--pos)" if ok and r.pnl >= 0 else "var(--neg)"),
            tip=(f"{r.tipo_valor} {r.serie}|{r.titulos:,.0f} títulos · "
                 f"valor {r.valor:,.0f}|DV01 {r.dv01:,.0f} MXN/pb · "
                 f"dur {r.DURACION:.2f}" if ok else
                 f"{r.tipo_valor} {r.serie}|No está en el vector vigente")))

    pos = dpo.posicionamiento_krd(dv, cart)
    max_k = float(max(pos["port_pct"].max(), pos["bench_pct"].max())) or 1.0
    krd_rows = [dict(
        cubeta=r.cubeta,
        wPort=f"{r.port_pct / max_k * 100:.0f}",
        wBench=f"{r.bench_pct / max_k * 100:.0f}",
        portF=f"{r.port_pct:.0f}%", benchF=f"{r.bench_pct:.0f}%",
        actF=f"{r.activo_pp:+.0f} pp",
        actC="var(--pos)" if r.activo_pp > 0 else "var(--neg)",
        tip=(f"Cubeta {r.cubeta}|Portafolio {r.port_pct:.1f} % del DV01 "
             f"({r.port_mxn:,.0f} MXN/pb)|Benchmark {r.bench_pct:.1f} % · "
             f"activo {r.activo_pp:+.1f} pp"))
        for r in pos.itertuples()]

    alm = dpo.alm_real(dv, cart)
    alm_kpis, alm_rows, repl_rows, alm_nota = [], [], [], ""
    if alm:
        cobertura = (alm["dv01_activo"] / alm["dv01_pasivo"] * 100.0
                     if alm["dv01_pasivo"] else 0.0)
        alm_kpis = [
            dict(label="PV pasivo real", value=f"${alm['pv_pasivo']/1e6:,.1f} M"),
            dict(label="Dur. real pasivo", value=f"{alm['dur_pasivo']:.1f}a"),
            dict(label="DV01 pasivo", value=f"{alm['dv01_pasivo']:,.0f}"),
            dict(label="DV01 activo real", value=f"{alm['dv01_activo']:,.0f}"),
            dict(label="Cobertura", value=f"{cobertura:.0f} %"),
        ]
        mx_a = float(alm["cubetas"][["pasivo", "activo"]].values.max()) or 1.0
        for r in alm["cubetas"].itertuples():
            alm_rows.append(dict(
                cubeta=r.cubeta,
                wPas=f"{r.pasivo / mx_a * 100:.0f}",
                wAct=f"{r.activo / mx_a * 100:.0f}",
                brechaF=f"{r.brecha:+,.0f}",
                brechaC="var(--pos)" if abs(r.brecha) < mx_a * 0.1 else "var(--neg)",
                tip=(f"Cubeta {r.cubeta}|KRD pasivo {r.pasivo:,.0f} · "
                     f"activo {r.activo:,.0f}|Brecha {r.brecha:+,.0f} MXN/pb")))
        for r in alm["replicante"].itertuples():
            repl_rows.append(dict(
                serie=str(r.serie), aniosF=f"{r.anios:.1f}a",
                realF=f"{r.real:.2f} %",
                invF=f"${r.inversion/1e6:,.1f} M"))
        alm_nota = (
            f"Pasivo: flujos reales anuales de data/pasivo_real.csv "
            f"descontados con la curva real de Udibonos del vector. La "
            f"cobertura de hoy es {cobertura:.0f} % del DV01 real del "
            f"pasivo; la brecha se concentra en las cubetas largas. La "
            f"cartera replicante reparte la inversión en Udibonos que "
            f"igualaría las KRD del pasivo (mínimos cuadrados, sin cortos).")

    ests = dpo.estructuras_dv01(dv)
    est_rows = []
    for e in ests:
        patas = [dict(sentido=p["sentido"], serie=p["serie"],
                      titF=f"{p['titulos']:,.0f}",
                      ytmF=f"{p['ytm']:.2f} %",
                      dv01F=f"{p['dv01']:+,.0f}",
                      sentC=("var(--pos)" if p["sentido"] == "Compra"
                             else "var(--warn)"))
                 for p in e["patas"]]
        est_rows.append(dict(
            nombre=e["nombre"], senal=e["senal"], patas=patas,
            netoF=f"DV01 neto {e['dv01_neto']:+,.0f}",
            carryF=f"carry+roll 3m {e['carry3m_neto']:+,.0f} MXN",
            carryC=("var(--pos)" if e["carry3m_neto"] >= 0 else "var(--neg)"),
            tip=(f"{e['nombre']}|DV01 bruto {e['dv01_bruto']:,.0f} MXN/pb · "
                 f"neto {e['dv01_neto']:+,.0f}|Carry+roll 3 meses "
                 f"{e['carry3m_neto']:+,.0f} MXN")))

    return dict(
        kpis=[
            dict(label="Valor cartera RF", value=f"${total/1e6:,.1f} M"),
            dict(label="DV01 total", value=f"{dv01_tot:,.0f} MXN/pb"),
            dict(label="Duración", value=f"{dur_pond:.2f}"),
            dict(label="Carry anual est.", value=f"{carry_anual:.2f} %"),
            dict(label="P&L vs costo", value=f"{pnl:+,.0f}"),
            dict(label="TE ex-ante",
                 value=(f"{te:.0f} pb" if te is not None
                        else f"— ({len(hist)} ses.)")),
        ],
        filas=filas, krdRows=krd_rows,
        almKpis=alm_kpis, almRows=alm_rows, replRows=repl_rows,
        almNota=alm_nota, estructuras=est_rows,
        notaKrd=("DV01 por cubeta clave (2/5/10/20/30a), repartido "
                 "linealmente entre las cubetas que flanquean cada "
                 "posición, contra el benchmark gubernamental de Bonos M "
                 "ponderado por monto en circulación del vector. La barra "
                 "morada es el portafolio; la gris, el benchmark."),
        notaCartera=("Cartera registrada en data/posiciones_deuda.csv "
                     "(boletas C/V neteadas con costo promedio) y valuada "
                     "con el PRECIO SUCIO oficial del vector vigente. El "
                     "TE ex-ante se activa al acumular 20 sesiones de "
                     "vectores. Análisis de mesa; no constituye una "
                     "recomendación de inversión."),
    )


def _seccion_cartera(html: str, pc: dict) -> str:
    # ---- navegacion, titulo y bandera ------------------------------------
    html = _sub(html,
        "['deuda-val','Valuación · Crédito','12'],"
        "['#','General'],['configuracion','Configuración','13']",
        "['deuda-val','Valuación · Crédito','12'],"
        "['deuda-port','Cartera · ALM','13'],"
        "['#','General'],['configuracion','Configuración','14']")
    html = _sub(html,
        "configuracion: ['Configuración', 'Fuentes de datos",
        "'deuda-port': ['Renta fija · Cartera y ALM', 'Posiciones propias, "
        "DV01 por cubeta vs benchmark, calce del pasivo real y estructuras "
        "DV01-neutral'], "
        "configuracion: ['Configuración', 'Fuentes de datos")
    html = _sub(html,
        "isDeudaVal: this.state.tab === 'deuda-val', ",
        "isDeudaVal: this.state.tab === 'deuda-val', "
        "isDeudaPort: this.state.tab === 'deuda-port', "
        "pcart: " + js(pc) + ", ")

    TD = '<\\u002Fdiv>'
    card = ('background:var(--surf);border:1px solid var(--border);'
            'border-radius:6px;padding:14px 16px')
    header = ('font:700 10px/1 var(--sans);letter-spacing:.14em;'
              'text-transform:uppercase;color:var(--ink3);margin-bottom:12px')
    nota = ('font:400 10.5px/1.55 var(--sans);color:var(--ink3);'
            'margin-top:10px;border-top:1px dashed var(--border);'
            'padding-top:8px')
    th = ('text-align:right;font:700 9px/1 var(--sans);'
          'letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);'
          'padding:8px 9px')
    th_izq = th.replace('text-align:right', 'text-align:left')
    td = 'padding:6px 9px;text-align:right;font-family:var(--mono)'

    seccion = (
        '<!-- ================= RF CARTERA ================= -->\\n'
        '      <sc-if value=\\"{{ isDeudaPort }}\\" '
        'hint-placeholder-val=\\"{{ false }}\\">\\n'
        # KPIs
        '      <div style=\\"display:grid;'
        'grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px\\">'
        '<sc-for list=\\"{{ pcart.kpis }}\\" as=\\"k\\" '
        'hint-placeholder-count=\\"6\\">'
        '<div style=\\"' + card.replace('padding:14px 16px',
                                        'padding:10px 12px') + '\\">'
        '<div style=\\"font:600 8.5px/1 var(--sans);letter-spacing:.1em;'
        'text-transform:uppercase;color:var(--ink3)\\">{{ k.label }}' + TD +
        '<div style=\\"font:600 14.5px/1.3 var(--mono);color:var(--ink);'
        'margin-top:4px\\">{{ k.value }}' + TD + TD + '<\\u002Fsc-for>' + TD
        + '\\n'
        # tabla de posiciones
        '      <div style=\\"' + card.replace('padding:14px 16px',
                                              'padding:0;overflow:hidden')
        + ';margin-bottom:14px\\">'
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Instrumento'
        '<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Títulos<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Costo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Sucio hoy<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Valor<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">YTM<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Dur.<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">DV01<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Peso<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">P&amp;L<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ pcart.filas }}\\" as=\\"f\\" '
        'hint-placeholder-count=\\"7\\">'
        '<sc-raw-tr data-tip=\\"{{ f.tip }}\\" '
        'style-hover=\\"background:var(--surf2)\\" '
        'style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:6px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ f.clase }} {{ f.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">{{ f.titF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink3)\\">'
        '{{ f.costoF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.sucioF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink)\\">'
        '{{ f.valorF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ f.ytmF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">{{ f.durF }}'
        '<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.dv01F }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ f.pesoF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:{{ f.pnlC }}\\">{{ f.pnlF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        # KRD vs benchmark | ALM
        '      <div style=\\"display:grid;grid-template-columns:1fr 1.2fr;'
        'gap:14px;margin-bottom:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">DV01 por cubeta vs benchmark'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · % del DV01 total<\\u002Fspan>' + TD +
        '<sc-for list=\\"{{ pcart.krdRows }}\\" as=\\"r\\" '
        'hint-placeholder-count=\\"5\\">'
        '<div data-tip=\\"{{ r.tip }}\\" style=\\"display:flex;'
        'align-items:center;gap:8px;margin-bottom:8.5px\\">'
        '<div style=\\"width:34px;font-family:var(--mono);font-size:10.5px;'
        'color:var(--ink2)\\">{{ r.cubeta }}' + TD +
        '<div style=\\"flex:1\\">'
        '<div style=\\"display:flex;align-items:center;gap:5px;'
        'margin-bottom:3px\\">'
        '<div style=\\"height:8px;width:{{ r.wPort }}%;'
        'background:var(--brand-lite);border-radius:2px\\">' + TD +
        '<span style=\\"font-family:var(--mono);font-size:9px;'
        'color:var(--ink3)\\">{{ r.portF }}<\\u002Fspan>' + TD +
        '<div style=\\"display:flex;align-items:center;gap:5px\\">'
        '<div style=\\"height:8px;width:{{ r.wBench }}%;'
        'background:var(--border2);border-radius:2px\\">' + TD +
        '<span style=\\"font-family:var(--mono);font-size:9px;'
        'color:var(--ink3)\\">{{ r.benchF }}<\\u002Fspan>' + TD + TD +
        '<div style=\\"width:52px;text-align:right;'
        'font-family:var(--mono);font-size:10px;font-weight:600;'
        'color:{{ r.actC }}\\">{{ r.actF }}' + TD + TD + '<\\u002Fsc-for>'
        '<div style=\\"' + nota + '\\">{{ pcart.notaKrd }}' + TD + TD + '\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Calce ALM real · activo vs pasivo'
        + TD +
        '<div style=\\"display:grid;grid-template-columns:repeat(5,1fr);'
        'gap:8px;margin-bottom:12px\\">'
        '<sc-for list=\\"{{ pcart.almKpis }}\\" as=\\"k\\" '
        'hint-placeholder-count=\\"5\\">'
        '<div style=\\"background:var(--surf2);border-radius:5px;'
        'padding:8px 10px\\">'
        '<div style=\\"font:600 8px/1 var(--sans);letter-spacing:.08em;'
        'text-transform:uppercase;color:var(--ink3)\\">{{ k.label }}' + TD +
        '<div style=\\"font:600 12.5px/1.3 var(--mono);color:var(--ink);'
        'margin-top:3px\\">{{ k.value }}' + TD + TD + '<\\u002Fsc-for>' + TD +
        '<sc-for list=\\"{{ pcart.almRows }}\\" as=\\"r\\" '
        'hint-placeholder-count=\\"5\\">'
        '<div data-tip=\\"{{ r.tip }}\\" style=\\"display:flex;'
        'align-items:center;gap:8px;margin-bottom:7.5px\\">'
        '<div style=\\"width:34px;font-family:var(--mono);font-size:10.5px;'
        'color:var(--ink2)\\">{{ r.cubeta }}' + TD +
        '<div style=\\"flex:1\\">'
        '<div style=\\"height:7px;width:{{ r.wPas }}%;'
        'background:var(--neg);opacity:.75;border-radius:2px;'
        'margin-bottom:3px\\">' + TD +
        '<div style=\\"height:7px;width:{{ r.wAct }}%;'
        'background:var(--pos);border-radius:2px\\">' + TD + TD +
        '<div style=\\"width:74px;text-align:right;'
        'font-family:var(--mono);font-size:9.5px;font-weight:600;'
        'color:{{ r.brechaC }}\\">{{ r.brechaF }}' + TD + TD +
        '<\\u002Fsc-for>'
        '<div style=\\"display:flex;gap:12px;font-family:var(--mono);'
        'font-size:9.5px;color:var(--ink3);margin-top:2px\\">'
        '<span>■ rojo: KRD pasivo<\\u002Fspan>'
        '<span style=\\"color:var(--pos)\\">■ verde: KRD activo real'
        '<\\u002Fspan>' + TD +
        '<div style=\\"' + nota + '\\">{{ pcart.almNota }}' + TD + TD + TD
        + '\\n'
        # replicante | estructuras
        '      <div style=\\"display:grid;grid-template-columns:1fr 1.6fr;'
        'gap:14px\\">\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Cartera replicante del pasivo'
        + TD +
        '<sc-raw-table style=\\"width:100%;border-collapse:collapse;'
        'font-size:10.5px\\"><sc-raw-thead><sc-raw-tr>'
        '<sc-raw-th style=\\"' + th_izq + '\\">Udibono<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Plazo<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Real<\\u002Fsc-raw-th>'
        '<sc-raw-th style=\\"' + th + '\\">Inversión<\\u002Fsc-raw-th>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-raw-thead><sc-raw-tbody>'
        '<sc-for list=\\"{{ pcart.replRows }}\\" as=\\"r\\" '
        'hint-placeholder-count=\\"13\\">'
        '<sc-raw-tr style=\\"border-top:1px solid var(--border)\\">'
        '<sc-raw-td style=\\"padding:5px 9px;font:600 10.5px var(--mono);'
        'color:var(--ink)\\">{{ r.serie }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ r.aniosF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';color:var(--ink2)\\">'
        '{{ r.realF }}<\\u002Fsc-raw-td>'
        '<sc-raw-td style=\\"' + td + ';font-weight:600;'
        'color:var(--ink)\\">{{ r.invF }}<\\u002Fsc-raw-td>'
        '<\\u002Fsc-raw-tr><\\u002Fsc-for>'
        '<\\u002Fsc-raw-tbody><\\u002Fsc-raw-table>' + TD + '\\n'
        '        <div style=\\"' + card + '\\">'
        '<div style=\\"' + header + '\\">Estructuras DV01-neutral'
        '<span style=\\"font-weight:500;text-transform:none;'
        'letter-spacing:0\\"> · pata principal 10,000 títulos'
        '<\\u002Fspan>' + TD +
        '<sc-for list=\\"{{ pcart.estructuras }}\\" as=\\"e\\" '
        'hint-placeholder-count=\\"3\\">'
        '<div data-tip=\\"{{ e.tip }}\\" style=\\"background:var(--surf2);'
        'border-radius:6px;padding:10px 12px;margin-bottom:9px\\">'
        '<div style=\\"display:flex;justify-content:space-between;'
        'align-items:baseline;gap:10px;flex-wrap:wrap\\">'
        '<div style=\\"font:600 11.5px/1.3 var(--sans);'
        'color:var(--ink)\\">{{ e.nombre }}' + TD +
        '<div style=\\"font:600 10.5px var(--mono);'
        'color:{{ e.carryC }}\\">{{ e.carryF }}' + TD + TD +
        '<div style=\\"font:400 10px/1.4 var(--sans);color:var(--ink3);'
        'margin:3px 0 7px\\">{{ e.senal }} · {{ e.netoF }}' + TD +
        '<sc-for list=\\"{{ e.patas }}\\" as=\\"p\\" '
        'hint-placeholder-count=\\"3\\">'
        '<div style=\\"display:flex;gap:10px;font-family:var(--mono);'
        'font-size:10.5px;padding:2px 0\\">'
        '<span style=\\"width:52px;font-weight:700;'
        'color:{{ p.sentC }}\\">{{ p.sentido }}<\\u002Fspan>'
        '<span style=\\"width:56px;color:var(--ink)\\">{{ p.serie }}'
        '<\\u002Fspan>'
        '<span style=\\"width:90px;text-align:right;'
        'color:var(--ink2)\\">{{ p.titF }} tít.<\\u002Fspan>'
        '<span style=\\"width:64px;text-align:right;'
        'color:var(--ink2)\\">{{ p.ytmF }}<\\u002Fspan>'
        '<span style=\\"flex:1;text-align:right;color:var(--ink3)\\">'
        'DV01 {{ p.dv01F }}<\\u002Fspan>' + TD + '<\\u002Fsc-for>' + TD +
        '<\\u002Fsc-for>'
        '<div style=\\"' + nota + '\\">{{ pcart.notaCartera }}' + TD + TD +
        TD + '\\n'
        '      <\\u002Fsc-if>\\n\\n      ')

    html = _sub(html,
        '<!-- ================= CONFIGURACIÓN ================= -->',
        seccion + '<!-- ================= CONFIGURACIÓN ================= -->')
    return html


def html_con_datos_reales(ruta_html: Path | None = None) -> str:
    ruta = ruta_html or (RAIZ / "assets" / "dashboard_baz.html")
    html = ruta.read_text(encoding="utf-8")
    resultado = inyectar(html, calcular())
    _verificar_bundle(resultado)
    return resultado
