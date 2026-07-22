"""
Punto de entrada unico para el despliegue en Streamlit Community Cloud.

Sirve el tablero institucional rediseñado con datos reales en vivo
(deploy/streamlit_dashboard.py). Es el archivo que Cloud detecta por
defecto, asi que el Main file path del despliegue es simplemente
`streamlit_app.py`.

app.py (el tablero operativo de la mesa) se conserva en el repo como
herramienta interna y no se despliega.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "deploy" / "streamlit_dashboard.py"),
    run_name="__main__",
)
