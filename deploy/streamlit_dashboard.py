"""
Dashboard de portafolio institucional — Punto Casa de Bolsa
============================================================

Embebe el tablero rediseñado (HTML autocontenido `dashboard_baz.html`) como
una app de Streamlit, lista para desplegar en Streamlit Community Cloud.

Ejecutar en local:
    streamlit run streamlit_dashboard.py

El tablero maneja su propio layout, scroll, tema claro/oscuro y navegación;
aquí solo se sirve a pantalla completa dentro de Streamlit.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Portafolio BAZ | Punto Casa de Bolsa",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Elimina el cromo y el padding por defecto de Streamlit para que el tablero
# aproveche todo el ancho y alto disponibles.
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

HTML = (Path(__file__).parent / "dashboard_baz.html").read_text(encoding="utf-8")

# El tablero ocupa 100vh internamente; el iframe le fija la altura y su propio
# scroll interno se encarga del contenido que exceda la ventana.
components.html(HTML, height=940, scrolling=True)
