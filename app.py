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


# ---------------------------------------------------------------------------
# Carga diaria del vector Valmer: el archivo subido releva al del repositorio
# (se resuelve por fecha de archivo) y las estrategias de renta fija se
# recalculan al momento. Vive fuera del tablero porque los botones del HTML
# no pueden alcanzar al servidor.
# ---------------------------------------------------------------------------
with st.expander("⬆ Actualizar vector Valmer (renta fija)", expanded=False):
    import hashlib

    from portfolio import bonos as _bonos

    try:
        _vec = _bonos.cargar()
        if _vec is not None:
            st.caption(
                f"Vector vigente: **{_vec['fecha']:%d %b %Y}** · "
                f"{len(_vec['cetes'])} CETES · {len(_vec['bonos_m'])} Bonos M · "
                f"{len(_vec['udibonos'])} Udibonos · "
                f"{len(_vec['bondesf'])} Bondes F")
        else:
            st.caption("Sin vector cargado todavía.")
    except Exception as e:
        st.caption(f"No se pudo leer el vector vigente ({type(e).__name__}).")

    _archivo = st.file_uploader(
        "Vector analítico del día (VectorAnaliticoMD.xls)",
        type=["xls", "xlsx"], key="vector_valmer",
        help="Al subirlo, la pestaña Deuda recalcula curvas, carry y "
             "estrategias con las cifras del día.")

    if _archivo is not None:
        _bytes = _archivo.getvalue()
        _sufijo = ".xlsx" if _archivo.name.lower().endswith("x") else ".xls"
        _destino = Path(__file__).parent / "data" / f"VectorSubido{_sufijo}"
        _md5 = hashlib.md5(_bytes).hexdigest()
        _previo = (hashlib.md5(_destino.read_bytes()).hexdigest()
                   if _destino.exists() else None)
        if _md5 != _previo:
            _destino.write_bytes(_bytes)
            try:
                _nuevo = _bonos.cargar(_destino)
                _val = _nuevo["validacion"]["max_pb"]
            except Exception as e:
                _destino.unlink(missing_ok=True)
                st.error(
                    f"El archivo no parece un vector analítico de Valmer "
                    f"({type(e).__name__}: {e}). No se aplicó ningún cambio.")
            else:
                _html_con_datos_reales.clear()
                st.success(
                    f"Vector del {_nuevo['fecha']:%d %b %Y} cargado: "
                    f"{len(_nuevo['bonos_m'])} Bonos M, "
                    f"{len(_nuevo['cetes'])} CETES, "
                    f"{len(_nuevo['udibonos'])} Udibonos. Verificación "
                    f"solver vs tasa oficial: {_val:.2f} pb. La pestaña "
                    f"Deuda ya muestra las estrategias del día.")

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
