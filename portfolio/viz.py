"""
Capa de graficos: plantilla Plotly y constructores de figuras.

Reglas de construccion aplicadas de forma consistente:
  - Paleta categorica de orden fijo, nunca ciclada ni generada.
  - Un solo eje de valor por figura; dos magnitudes distintas se indizan
    a una base comun o se separan en dos graficos.
  - Marcas delgadas, extremos redondeados de 4 px, separacion de 2 px
    entre rellenos contiguos.
  - Rejilla y ejes recesivos; etiquetas directas cuando hay 4 series o menos.
  - Capa de hover siempre presente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------
# Temas
# --------------------------------------------------------------------------
# Ambos modos usan las mismas ocho familias de color categorico en el mismo
# orden fijo; cada modo toma el escalon validado para su superficie (banda de
# luminosidad, piso de croma, separacion CVD adyacente >= 8.4, piso de vision
# normal >= 19.3 y contraste >= 3:1 en las ocho ranuras). Los tonos de cifra
# (pos/neg de texto) estan verificados a >= 4.5:1 sobre su superficie; los de
# marca (rellenos) a >= 3:1.

TEMAS: dict[str, dict] = {
    "dark": dict(
        plano="#0a0a0f", superficie="#14141b", borde="#2a2a38",
        tinta="#f4f3f7", tinta_2="#a8a5b8", tinta_3="#7d7a90",
        marca="#522D6D", acento="#9085e9",
        positivo="#0ca30c", negativo="#d03b3b",
        positivo_txt="#22c55e", negativo_txt="#ef4444",
        neutro="#4a4a5c", neutro_medio="#6b6b78",
        treemap_txt="#ffffff",
        categorica=["#3987e5", "#008300", "#d55181", "#c98500",
                    "#199e70", "#d95926", "#9085e9", "#e66767"],
        divergente_frio=("#1b4f8f", "#3987e5"),
        divergente_calido=("#d95926", "#8f2f10"),
    ),
    "light": dict(
        plano="#f7f6f9", superficie="#fcfcfb", borde="#e0dee8",
        tinta="#17161f", tinta_2="#4d4a5a", tinta_3="#716e82",
        marca="#522D6D", acento="#5e3a86",
        positivo="#0ca30c", negativo="#d03b3b",
        positivo_txt="#006300", negativo_txt="#b3281e",
        neutro="#c6c4d2", neutro_medio="#d8d6e0",
        treemap_txt="#ffffff",
        categorica=["#2a78d6", "#008300", "#e87ba4", "#eda100",
                    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"],
        divergente_frio=("#104281", "#2a78d6"),
        divergente_calido=("#eb6834", "#8f2f10"),
    ),
}

# Globales del tema activo; los constructores de figuras los leen al vuelo.
MODO = "dark"
PLANO = SUPERFICIE = BORDE = TINTA = TINTA_2 = TINTA_3 = ""
MARCA = MARCA_CLARA = POSITIVO = NEGATIVO = ""
POSITIVO_TXT = NEGATIVO_TXT = NEUTRO = NEUTRO_MEDIO = TREEMAP_TXT = ""
CATEGORICA: list[str] = []
_DIV_FRIO = _DIV_CALIDO = ("", "")


def activar_tema(modo: str = "dark") -> None:
    """Fija los globales de color al tema pedido ('dark' o 'light')."""
    global MODO, PLANO, SUPERFICIE, BORDE, TINTA, TINTA_2, TINTA_3
    global MARCA, MARCA_CLARA, POSITIVO, NEGATIVO, POSITIVO_TXT, NEGATIVO_TXT
    global NEUTRO, NEUTRO_MEDIO, TREEMAP_TXT, CATEGORICA, _DIV_FRIO, _DIV_CALIDO

    t = TEMAS.get(modo, TEMAS["dark"])
    MODO = modo if modo in TEMAS else "dark"
    PLANO, SUPERFICIE, BORDE = t["plano"], t["superficie"], t["borde"]
    TINTA, TINTA_2, TINTA_3 = t["tinta"], t["tinta_2"], t["tinta_3"]
    MARCA, MARCA_CLARA = t["marca"], t["acento"]
    POSITIVO, NEGATIVO = t["positivo"], t["negativo"]
    POSITIVO_TXT, NEGATIVO_TXT = t["positivo_txt"], t["negativo_txt"]
    NEUTRO, NEUTRO_MEDIO = t["neutro"], t["neutro_medio"]
    TREEMAP_TXT = t["treemap_txt"]
    CATEGORICA = list(t["categorica"])
    _DIV_FRIO, _DIV_CALIDO = t["divergente_frio"], t["divergente_calido"]


activar_tema("dark")

MONO = "JetBrains Mono, IBM Plex Mono, SF Mono, Consolas, monospace"
SANS = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"


def registrar_plantilla(modo: str = "dark") -> None:
    """Activa el tema pedido y registra la plantilla 'punto' en Plotly."""
    activar_tema(modo)
    pio.templates["punto"] = go.layout.Template(
        layout=dict(
            paper_bgcolor=SUPERFICIE,
            plot_bgcolor=SUPERFICIE,
            font=dict(family=SANS, size=11.5, color=TINTA_2),
            title=dict(font=dict(family=SANS, size=13, color=TINTA), x=0, xanchor="left"),
            colorway=CATEGORICA,
            xaxis=dict(gridcolor=BORDE, zerolinecolor=BORDE, linecolor=BORDE,
                       tickfont=dict(family=MONO, size=10, color=TINTA_3),
                       showgrid=False, automargin=True),
            yaxis=dict(gridcolor=BORDE, zerolinecolor=BORDE, linecolor=BORDE,
                       tickfont=dict(family=MONO, size=10, color=TINTA_3),
                       gridwidth=1, automargin=True),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10.5, color=TINTA_2),
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            hoverlabel=dict(
                bgcolor="#1c1c26" if MODO == "dark" else "#ffffff",
                bordercolor=BORDE,
                font=dict(family=MONO, size=11, color=TINTA)),
            margin=dict(l=10, r=14, t=34, b=10),
            separators=".,",
        )
    )
    pio.templates.default = "punto"


def _color_direccional(valores) -> list[str]:
    """Verde/rojo segun signo. Se acompana siempre de etiqueta o eje con signo."""
    return [POSITIVO if v >= 0 else NEGATIVO for v in valores]


def _vacio(mensaje: str = "Sin datos suficientes") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False,
                       font=dict(family=SANS, size=12, color=TINTA_3))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=240)
    return fig


# --------------------------------------------------------------------------
# Composicion
# --------------------------------------------------------------------------

def treemap_composicion(valuada: pd.DataFrame, dimension: str = "sector",
                        altura: int = 430) -> go.Figure:
    """
    Mapa de arbol: tamano = valor de mercado, color = dimension (identidad).
    Las etiquetas directas son el codificado secundario que exige la
    validacion de la paleta en formas de todos-contra-todos.
    """
    if not len(valuada):
        return _vacio()

    df = valuada.copy()
    niveles = df[dimension].astype(str)
    orden = list(dict.fromkeys(niveles))
    mapa = {n: CATEGORICA[i % len(CATEGORICA)] for i, n in enumerate(orden)}

    # Plotly exige que cada valor de `parents` exista tambien como `labels`.
    # Sin los nodos padre explicitos el arbol queda huerfano y no dibuja nada.
    etiquetas = orden + df["emisora"].astype(str).tolist()
    padres = [""] * len(orden) + niveles.tolist()
    valores = [0.0] * len(orden) + df["valor_mercado"].tolist()
    pesos = [df.loc[niveles == n, "peso_pct"].sum() for n in orden] + \
            df["peso_pct"].tolist()
    colores = [mapa[n] for n in orden] + [mapa[n] for n in niveles]

    fig = go.Figure(go.Treemap(
        labels=etiquetas, parents=padres, values=valores,
        branchvalues="remainder",   # el padre suma lo de sus hijos
        marker=dict(colors=colores,
                    line=dict(color=SUPERFICIE, width=2)),  # separacion de 2 px
        customdata=[f"{p:.1f} %" for p in pesos],
        texttemplate="<b>%{label}</b><br>%{customdata}",
        textfont=dict(family=MONO, size=10.5, color=TREEMAP_TXT),
        hovertemplate=("<b>%{label}</b><br>Valor  %{value:,.0f} MXN"
                       "<br>Peso   %{customdata}<extra></extra>"),
        tiling=dict(pad=2),
        pathbar=dict(visible=False),
    ))
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=8, b=0))
    return fig


def _altura_por_filas(n: int, minimo: int = 300, maximo: int = 780) -> int:
    """
    Alto del lienzo en funcion del numero de barras horizontales. Con altura
    fija, una dimension de muchas categorias (industria tiene 23) comprimia
    las barras hasta volverlas ilegibles.
    """
    return int(max(minimo, min(maximo, 110 + 28 * n)))


def barras_dimension(agrupado: pd.DataFrame, dimension: str,
                     altura: int | None = None) -> go.Figure:
    """
    Peso por dimension. Categoria nominal sin orden intrinseco: una sola
    serie, por lo tanto un solo tono y sin caja de leyenda.
    """
    if not len(agrupado):
        return _vacio()

    altura = altura or _altura_por_filas(len(agrupado))
    df = agrupado.sort_values("valor_mercado")
    fig = go.Figure(go.Bar(
        x=df["peso_pct"], y=df[dimension].astype(str), orientation="h",
        marker=dict(color=CATEGORICA[0],
                    line=dict(color=SUPERFICIE, width=2)),
        text=[f"{p:.1f}%" for p in df["peso_pct"]],
        textposition="outside",
        textfont=dict(family=MONO, size=10, color=TINTA_2),
        customdata=np.stack([df["valor_mercado"], df["no_realizado"],
                             df["rend_pct"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Peso    %{x:.2f} %"
                       "<br>Valor   %{customdata[0]:,.0f} MXN"
                       "<br>P&L     %{customdata[1]:,.0f} MXN"
                       "<br>Rend.   %{customdata[2]:+.2f} %<extra></extra>"),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=altura, bargap=0.34,
        xaxis=dict(title="Peso del portafolio (%)", showgrid=True, ticksuffix=" %"),
        yaxis=dict(title=None),
        margin=dict(l=6, r=54, t=12, b=8),
    )
    return fig


# --------------------------------------------------------------------------
# Resultado y atribucion
# --------------------------------------------------------------------------

def barras_contribucion(df: pd.DataFrame, etiqueta: str, valor: str,
                        titulo_eje: str = "Resultado no realizado (MXN)",
                        altura: int | None = None, n: int = 18,
                        formato: str = "mxn",
                        col_monto: str | None = None) -> go.Figure:
    """
    Aporte al resultado, ordenado. Polaridad -> escala divergente con
    gris neutro implicito en el cero; el eje con signo es el codificado
    secundario que evita depender solo del color.

    Con formato='pp' el valor viene en puntos porcentuales de contribucion al
    rendimiento del portafolio (la convencion institucional para comparar
    aportes entre posiciones de distinto tamano); `col_monto` agrega el
    importe en pesos al hover para no perder la magnitud absoluta.
    """
    if not len(df):
        return _vacio()

    d = df.reindex(df[valor].abs().sort_values(ascending=False).index).head(n)
    d = d.sort_values(valor)
    altura = altura or _altura_por_filas(len(d))

    if formato == "pp":
        texto = [f"{v:+.2f}" for v in d[valor]]
        sufijo_eje = dict(ticksuffix=" pp")
        if col_monto and col_monto in d.columns:
            hover = ("<b>%{y}</b><br>Contribucion  %{x:+.2f} pp"
                     "<br>Importe       %{customdata:,.0f} MXN<extra></extra>")
            customdata = d[col_monto]
        else:
            hover = "<b>%{y}</b><br>Contribucion  %{x:+.2f} pp<extra></extra>"
            customdata = None
    else:
        texto = [f"{v/1e6:+,.2f} M" for v in d[valor]]
        sufijo_eje = {}
        hover = "<b>%{y}</b><br>" + titulo_eje + "  %{x:,.0f}<extra></extra>"
        customdata = None

    fig = go.Figure(go.Bar(
        x=d[valor], y=d[etiqueta].astype(str), orientation="h",
        marker=dict(color=_color_direccional(d[valor]),
                    line=dict(color=SUPERFICIE, width=2)),
        text=texto,
        textposition="outside",
        textfont=dict(family=MONO, size=10, color=TINTA_2),
        customdata=customdata,
        hovertemplate=hover,
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line=dict(color=TINTA_3, width=1))
    fig.update_layout(
        height=altura, bargap=0.32,
        xaxis=dict(title=titulo_eje, showgrid=True, **sufijo_eje),
        yaxis=dict(title=None),
        margin=dict(l=6, r=76, t=12, b=8),
    )
    return fig


def linea_desempeno(serie_port: pd.Series, serie_bench: pd.Series | None = None,
                    nombre_bench: str = "IPC", altura: int = 380) -> go.Figure:
    """
    Portafolio contra referencia, ambos indizados a 100 en el origen.
    La indizacion es lo que permite una sola escala: nunca dos ejes.
    Dos series -> leyenda presente y ademas etiqueta directa al final.
    """
    if serie_port is None or len(serie_port) < 2:
        return _vacio("Sin historico suficiente")

    base_p = serie_port.iloc[0]
    idx_p = serie_port / base_p * 100.0

    fig = go.Figure()
    # Sin relleno al eje: un indice no es una magnitud medida desde cero, y
    # anclar el area en cero obligaria al eje a abarcar 0-120, aplastando la
    # variacion real contra el borde superior.
    fig.add_trace(go.Scatter(
        x=idx_p.index, y=idx_p.values, name="Portafolio",
        mode="lines", line=dict(color=MARCA_CLARA, width=2),
        hovertemplate="<b>Portafolio</b><br>%{x|%d %b %Y}<br>Base 100  %{y:.2f}<extra></extra>",
    ))
    fig.add_annotation(x=idx_p.index[-1], y=idx_p.iloc[-1], text="Portafolio",
                       showarrow=False, xanchor="left", xshift=7,
                       font=dict(family=SANS, size=10.5, color=MARCA_CLARA))

    if serie_bench is not None and len(serie_bench) >= 2:
        b = serie_bench.reindex(serie_port.index).ffill().dropna()
        if len(b) >= 2:
            idx_b = b / b.iloc[0] * 100.0
            fig.add_trace(go.Scatter(
                x=idx_b.index, y=idx_b.values, name=nombre_bench,
                mode="lines", line=dict(color=TINTA_3, width=2, dash="dot"),
                hovertemplate="<b>" + nombre_bench +
                              "</b><br>%{x|%d %b %Y}<br>Base 100  %{y:.2f}<extra></extra>",
            ))
            fig.add_annotation(x=idx_b.index[-1], y=idx_b.iloc[-1], text=nombre_bench,
                               showarrow=False, xanchor="left", xshift=7,
                               font=dict(family=SANS, size=10.5, color=TINTA_3))

    fig.add_hline(y=100, line=dict(color=BORDE, width=1, dash="dash"))
    fig.update_layout(
        height=altura, hovermode="x unified",
        yaxis=dict(title="Indice (base 100)", showgrid=True, autorange=True,
                   rangemode="normal"),
        xaxis=dict(showgrid=False),
        margin=dict(l=6, r=76, t=30, b=8),
    )
    return fig


def area_drawdown(serie_port: pd.Series, altura: int = 250) -> go.Figure:
    """Caida acumulada desde el maximo previo."""
    if serie_port is None or len(serie_port) < 2:
        return _vacio("Sin historico suficiente")
    dd = (serie_port / serie_port.cummax() - 1.0) * 100.0
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, mode="lines",
        line=dict(color=NEGATIVO, width=2),
        fill="tozeroy", fillcolor="rgba(208,59,59,0.18)",
        hovertemplate="%{x|%d %b %Y}<br>Caida  %{y:.2f} %<extra></extra>",
    ))
    fig.update_layout(
        height=altura, hovermode="x unified",
        yaxis=dict(title="Caida desde maximo (%)", ticksuffix=" %", showgrid=True),
        xaxis=dict(showgrid=False), margin=dict(l=6, r=14, t=12, b=8),
    )
    return fig


# --------------------------------------------------------------------------
# Efectivo
# --------------------------------------------------------------------------

def escalones_efectivo(serie: pd.DataFrame, altura: int = 300) -> go.Figure:
    """Saldo de efectivo en escalones, con los flujos del dia en el hover."""
    if not len(serie):
        return _vacio("Aun no hay movimientos registrados")

    fig = go.Figure(go.Scatter(
        x=serie["fecha"], y=serie["efectivo"], mode="lines+markers",
        line=dict(color=CATEGORICA[4], width=2, shape="hv"),
        marker=dict(size=8, color=CATEGORICA[4],
                    line=dict(color=SUPERFICIE, width=2)),  # anillo de 2 px
        customdata=serie["flujo"],
        hovertemplate=("%{x|%d %b %Y}<br>Saldo   %{y:,.0f} MXN"
                       "<br>Flujo   %{customdata:+,.0f} MXN<extra></extra>"),
    ))
    fig.add_hline(y=0, line=dict(color=TINTA_3, width=1, dash="dash"))
    fig.update_layout(
        height=altura, hovermode="x unified",
        yaxis=dict(title="Saldo de efectivo (MXN)", showgrid=True),
        xaxis=dict(showgrid=False), margin=dict(l=6, r=14, t=12, b=8),
    )
    return fig


def barras_flujo(bitacora: pd.DataFrame, altura: int = 280) -> go.Figure:
    """Entradas y salidas de efectivo por dia."""
    if not len(bitacora) or bitacora["fecha"].isna().all():
        return _vacio("Aun no hay movimientos registrados")

    b = bitacora.dropna(subset=["fecha"])
    g = b.groupby("fecha", as_index=False)["efecto_caja"].sum()

    fig = go.Figure(go.Bar(
        x=g["fecha"], y=g["efecto_caja"],
        marker=dict(color=_color_direccional(g["efecto_caja"]),
                    line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="%{x|%d %b %Y}<br>Flujo neto  %{y:,.0f} MXN<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=TINTA_3, width=1))
    fig.update_layout(
        height=altura, bargap=0.5,
        yaxis=dict(title="Flujo neto de efectivo (MXN)", showgrid=True),
        xaxis=dict(showgrid=False), margin=dict(l=6, r=14, t=12, b=8),
    )
    return fig


# --------------------------------------------------------------------------
# Riesgo
# --------------------------------------------------------------------------

def barras_riesgo_vs_peso(riesgo_pos: pd.DataFrame, altura: int = 430,
                          n: int = 15) -> go.Figure:
    """
    Peso contra contribucion al riesgo. Ambas magnitudes son porcentajes
    del total, por lo que comparten escala legitimamente: dos series
    agrupadas, con leyenda.
    """
    if not len(riesgo_pos):
        return _vacio("Se requiere historico de al menos 30 sesiones")

    d = riesgo_pos.head(n).sort_values("contrib_riesgo_pct")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["peso_pct"], y=d["emisora"].astype(str), orientation="h",
        name="Peso en cartera", marker=dict(color=NEUTRO,
                                            line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="<b>%{y}</b><br>Peso  %{x:.2f} %<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=d["contrib_riesgo_pct"], y=d["emisora"].astype(str), orientation="h",
        name="Contribucion al riesgo",
        marker=dict(color=CATEGORICA[5], line=dict(color=SUPERFICIE, width=2)),
        hovertemplate="<b>%{y}</b><br>Riesgo  %{x:.2f} %<extra></extra>",
    ))
    fig.update_layout(
        height=altura, barmode="group", bargap=0.28, bargroupgap=0.12,
        xaxis=dict(title="Porcentaje del total (%)", ticksuffix=" %", showgrid=True),
        yaxis=dict(title=None), margin=dict(l=6, r=14, t=30, b=8),
    )
    return fig


def mapa_correlacion(corr: pd.DataFrame, altura: int = 500) -> go.Figure:
    """
    Correlaciones. Polaridad en torno a cero -> escala divergente de dos
    tonos con gris neutro en el punto medio, nunca un arcoiris.
    """
    if not len(corr):
        return _vacio("Se requiere historico de al menos 30 sesiones")

    escala = [
        [0.00, _DIV_FRIO[0]], [0.25, _DIV_FRIO[1]], [0.50, NEUTRO_MEDIO],
        [0.75, _DIV_CALIDO[0]], [1.00, _DIV_CALIDO[1]],
    ]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=escala, zmid=0, zmin=-1, zmax=1,
        xgap=2, ygap=2,  # separacion de 2 px entre celdas
        colorbar=dict(title=dict(text="Corr.", font=dict(size=10)),
                      tickfont=dict(family=MONO, size=9.5), thickness=11,
                      outlinewidth=0, len=0.85),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlacion  %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=altura,
        xaxis=dict(tickangle=-45, tickfont=dict(family=MONO, size=9.5), showgrid=False),
        yaxis=dict(tickfont=dict(family=MONO, size=9.5), showgrid=False,
                   autorange="reversed"),
        margin=dict(l=6, r=6, t=12, b=8),
    )
    return fig


def dispersion_riesgo_rendimiento(valuada: pd.DataFrame, tecnicos: pd.DataFrame,
                                  riesgo_pos: pd.DataFrame,
                                  altura: int = 440) -> go.Figure:
    """
    Volatilidad individual contra rendimiento acumulado; el tamano codifica
    el peso en cartera. Al ser una forma de todos-contra-todos, la identidad
    se resuelve con etiqueta directa y no con color.
    """
    if not len(valuada) or not len(riesgo_pos):
        return _vacio("Se requiere historico de al menos 30 sesiones")

    d = valuada.merge(riesgo_pos[["ticker", "vol_individual_pct"]],
                      on="ticker", how="inner")
    if not len(d):
        return _vacio()

    tam = 9 + (d["peso_pct"] / d["peso_pct"].max()) * 34

    fig = go.Figure(go.Scatter(
        x=d["vol_individual_pct"], y=d["rend_pct"], mode="markers+text",
        text=d["emisora"], textposition="top center",
        textfont=dict(family=MONO, size=8.6, color=TINTA_3),
        marker=dict(size=tam, color=_color_direccional(d["rend_pct"]),
                    opacity=0.82,
                    line=dict(color=SUPERFICIE, width=2)),  # anillo de 2 px
        customdata=np.stack([d["peso_pct"], d["valor_mercado"],
                             d["no_realizado"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>Volatilidad  %{x:.1f} %"
                       "<br>Rendimiento  %{y:+.1f} %"
                       "<br>Peso         %{customdata[0]:.2f} %"
                       "<br>P&L          %{customdata[2]:,.0f} MXN<extra></extra>"),
    ))
    fig.add_hline(y=0, line=dict(color=TINTA_3, width=1, dash="dash"))
    fig.update_layout(
        height=altura,
        xaxis=dict(title="Volatilidad anualizada (%)", ticksuffix=" %", showgrid=True),
        yaxis=dict(title="Rendimiento acumulado (%)", ticksuffix=" %", showgrid=True),
        margin=dict(l=6, r=14, t=12, b=8),
    )
    return fig


# --------------------------------------------------------------------------
# Atribucion
# --------------------------------------------------------------------------

def cascada_atribucion(bf: dict, altura: int = 360) -> go.Figure:
    """
    Cascada de Brinson-Fachler: asignacion + seleccion + interaccion suman
    exactamente el retorno activo, y la cascada hace visible esa identidad.
    """
    if not bf or bf.get("activo") != bf.get("activo"):
        return _vacio("Sin datos suficientes para la atribucion")

    valores = [bf["asignacion"], bf["seleccion"], bf["interaccion"]]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Asignación", "Selección", "Interacción", "Retorno activo"],
        y=valores + [0.0],
        text=[f"{v:+.2f}" for v in valores + [bf["activo"]]],
        textposition="outside",
        textfont=dict(family=MONO, size=11, color=TINTA_2),
        connector=dict(line=dict(color=BORDE, width=1)),
        increasing=dict(marker=dict(color=POSITIVO)),
        decreasing=dict(marker=dict(color=NEGATIVO)),
        totals=dict(marker=dict(color=MARCA_CLARA)),
        hovertemplate="<b>%{x}</b><br>%{y:+.2f} pp<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=TINTA_3, width=1))
    fig.update_layout(
        height=altura, showlegend=False,
        yaxis=dict(title="Puntos porcentuales", ticksuffix=" pp", showgrid=True),
        xaxis=dict(showgrid=False),
        margin=dict(l=6, r=14, t=26, b=8),
    )
    return fig


def linea_movil(serie: pd.Series, titulo_eje: str, color: str | None = None,
                referencia: float | None = None,
                altura: int = 300) -> go.Figure:
    """Serie movil (volatilidad o beta) con linea de referencia opcional."""
    if serie is None or len(serie) < 5:
        return _vacio("Historico insuficiente para la serie movil")

    fig = go.Figure(go.Scatter(
        x=serie.index, y=serie.values, mode="lines",
        line=dict(color=color or CATEGORICA[0], width=2),
        hovertemplate="%{x|%d %b %Y}<br>" + titulo_eje + "  %{y:.2f}<extra></extra>",
    ))
    if referencia is not None:
        fig.add_hline(y=referencia,
                      line=dict(color=TINTA_3, width=1, dash="dash"))
    fig.update_layout(
        height=altura, hovermode="x unified",
        yaxis=dict(title=titulo_eje, showgrid=True),
        xaxis=dict(showgrid=False),
        margin=dict(l=6, r=14, t=12, b=8),
    )
    return fig
