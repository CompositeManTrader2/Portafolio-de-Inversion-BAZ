# Despliegue del tablero en Streamlit

Este paquete contiene el tablero **Dashboard de portafolio institucional**
(Banco Azteca · Contrato 104351) listo para correr y desplegar en Streamlit.

```
deploy/
├── dashboard_baz.html       ← tablero completo, HTML autocontenido (un solo archivo)
├── streamlit_dashboard.py   ← wrapper que lo sirve como app de Streamlit
└── README_DEPLOY.md         ← este archivo
```

El HTML es **autocontenido**: no necesita internet ni archivos externos
(estilos, fuentes y runtime van embebidos). Streamlit solo lo muestra a
pantalla completa.

---

## 1. Correr en local

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt        # streamlit ya está en la lista
streamlit run deploy/streamlit_dashboard.py
```

Abre `http://localhost:8501`.

---

## 2. Subirlo al repositorio

1. Copia la carpeta `deploy/` a la raíz de tu repo **Portafolio de BAZ**.
2. Confirma que `streamlit` esté en `requirements.txt` (ya lo está).
3. Commit y push:

   ```bash
   git add deploy/
   git commit -m "Agrega tablero rediseñado (Streamlit + HTML)"
   git push
   ```

> No sobrescribas tu `app.py` actual: ese es el tablero con datos reales.
> Este `streamlit_dashboard.py` es una app separada (la vista rediseñada).

---

## 3. Desplegar en Streamlit Community Cloud

1. Entra a <https://share.streamlit.io> y conecta tu cuenta de GitHub.
2. **New app** → elige el repo y la rama.
3. En **Main file path** escribe:

   ```
   deploy/streamlit_dashboard.py
   ```

4. **Deploy**. En ~1 min tendrás la URL pública.

---

## 4. Actualizar el tablero después de un cambio de diseño

El HTML se genera a partir del componente de diseño `Dashboard Portafolios.dc.html`.
Cuando lo modifiques, vuelve a exportarlo como HTML autocontenido y reemplaza
`deploy/dashboard_baz.html`.

---

## 5. (Opcional) Integrarlo como pestaña en tu `app.py`

Si prefieres que viva dentro de tu app actual en vez de ser una app aparte,
agrega una pestaña y embébelo:

```python
from pathlib import Path
import streamlit.components.v1 as components

# ... dentro de tus st.tabs([...]) agrega "Tablero" y en su bloque:
html = (Path(__file__).parent / "deploy" / "dashboard_baz.html").read_text(encoding="utf-8")
components.html(html, height=940, scrolling=True)
```

---

## Nota sobre los datos

El tablero muestra **datos de ejemplo** coherentes con la taxonomía del repo
(30 emisoras, sectores, IPC como benchmark). Para conectarlo a las cifras
reales que ya calcula `app.py`, el siguiente paso es inyectar esos valores al
HTML (por ejemplo, serializando los datos a JSON e insertándolos antes de
servir el componente). Puedo ayudarte con esa integración cuando quieras.
