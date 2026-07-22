"""
Portafolio de Inversión BAZ — Punto Casa de Bolsa
==================================================

Tablero institucional de seguimiento diario. La vista es el rediseño HTML
autocontenido (assets/dashboard_baz.html); antes de servirlo, se le inyectan
los datos reales calculados por el paquete `portfolio` (posición base +
boletas del custodio en data/ + precios vivos de Yahoo + analítica completa,
incluida la atribución Brinson-Fachler por periodo).

Ejecutar en local:
    streamlit run app.py

Despliegue en Streamlit Community Cloud: Main file path = app.py

Para integrar operaciones nuevas basta con dejar la boleta del custodio
(u otro Excel con Fecha/Operación/Emisora/Títulos/Precio) en la carpeta
data/ y hacer push: la app la concilia sin duplicar.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Portafolio de Inversión BAZ | Punto Casa de Bolsa",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# El tablero maneja su propio layout y navegación; se elimina el cromo y el
# padding de Streamlit para que use todo el ancho y alto disponibles.
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding: 0 !important; max-width: 100% !important;}
      section.main > div {padding: 0 !important;}
      iframe {border: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner="Calculando el portafolio con precios de mercado…")
def _html_con_datos_reales() -> str:
    """
    HTML del tablero con las cifras reales inyectadas. Se memoriza cinco
    minutos para no golpear a Yahoo en cada interacción; el botón de
    recarga del navegador (o la tecla R) fuerza el recálculo al expirar.
    """
    from portfolio.datos_reales import html_con_datos_reales
    return html_con_datos_reales()


try:
    HTML = _html_con_datos_reales()
except Exception as e:
    # Sin red o con el pipeline roto es mejor mostrar la plantilla de
    # ejemplo con una advertencia visible que una pantalla de error.
    HTML = (Path(__file__).parent / "assets" / "dashboard_baz.html").read_text(
        encoding="utf-8")
    st.warning(
        f"No fue posible calcular los datos reales "
        f"({type(e).__name__}: {e}). Se muestra el tablero con datos de "
        f"ejemplo; recarga la página para reintentar.")

# El tablero ocupa 100vh internamente; el iframe fija la altura y el scroll
# interno se encarga del contenido que exceda la ventana.
components.html(HTML, height=940, scrolling=True)
