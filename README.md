# Portafolio de Inversión BAZ

Tablero interactivo de seguimiento diario para la cartera institucional de
**Banco Azteca (contrato 104351)**, desarrollado para **Punto Casa de Bolsa**.

Reconstruye la posición vigente a partir de una foto base más los movimientos
subsecuentes, la valúa contra precios de mercado, y produce el bloque de
atribución, riesgo y diagnóstico que se presenta al cliente.

---

## Qué resuelve

| | |
|---|---|
| **Posición** | Aplica los movimientos sobre la foto base con costo promedio ponderado |
| **Efectivo** | Deriva el saldo de los flujos de compra y venta, con comisión e IVA |
| **Valuación** | Precios en vivo de las 30 emisoras vía Yahoo Finance |
| **Segmentación** | Sector, industria, región, clase de activo, mercado y divisa |
| **Riesgo** | Volatilidad, Sharpe, Sortino, beta, VaR/CVaR, drawdown, correlaciones |
| **Atribución** | Contribución al resultado por dimensión y por emisora |
| **Diagnóstico** | Reglas de mesa que señalan qué está funcionando y qué no |
| **Captura** | Alta de operaciones desde la propia interfaz |

---

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

Al arrancar toma automáticamente el archivo de posición que esté en `data/`.
También se puede cargar otro desde el panel lateral.

---

## Modelo de datos

### Posición base
Hoja con `Emisora`, `Títulos` y `Precio Compra`. Los montos se recalculan
siempre como `títulos × precio`: el archivo original los expresa en millones y
con redondeos, así que recomputarlos evita arrastrar esa pérdida de precisión.

### Movimientos
Se consolidan de tres fuentes, sin duplicar:

1. **Bitácora interna** — hoja `Coste` del mismo archivo.
2. **Archivos de operaciones** — todo `.xlsx` en `data/` que no sea el de
   posición, más los que se carguen desde el panel lateral. Se recorren todas
   sus hojas. Las boletas del custodio aportan además comisión e IVA reales.
3. **Captura manual** — desde la pestaña *Operaciones*.

`data/` incluye las boletas del **16, 17, 20 y 21 de julio de 2026**, que
cubren las 10 operaciones del periodo con sus costos reales.

Una operación se identifica por `(emisora, sentido, títulos, precio)`. La fecha
queda deliberadamente fuera de la llave porque la bitácora interna y la boleta
pueden diferir —concertación contra liquidación— o venir vacía; al consolidar
se conserva el primer valor no nulo de cada campo, de modo que la boleta aporta
los costos reales y la bitácora la fecha.

### Reglas de costeo

```
COMPRA   títulos += q
         costo_total += q × precio
         precio_costo = costo_total / títulos      (promedio ponderado)
         efectivo -= (q × precio + comisión + IVA)

VENTA    títulos -= q
         precio_costo NO se modifica
         costo_total = títulos × precio_costo       (se reduce a prorrata)
         efectivo += (q × precio − comisión − IVA)
         realizado = q × (precio − precio_costo) − comisión − IVA
```

Los costos de operación están calibrados contra la boleta `Res.104351`:
**comisión de 4 pb sobre el monto bruto e IVA de 16 % sobre la comisión**.
Verificado contra la boleta: ALSEA 50,000 @ 41.44858 → bruto 2,072,429.00,
comisión 828.9716, IVA 132.635451, neto 2,073,390.607051.

La reconstrucción reproduce exactamente las cifras del archivo fuente
(AC \* → 150,231 títulos a 201.98036133734715, costo 30,343,711.664071).

---

## Precios

Las 30 emisoras se mapean a tickers `.MX` de Yahoo Finance. **Todas cotizan en
pesos, incluidas las del SIC**, por lo que la valuación es directa y no requiere
conversión cambiaria. El tipo de cambio se descarga sólo para dimensionar la
exposición económica al dólar, que sí existe: 40 % del portafolio tiene valor
subyacente en USD.

Algunas emisoras del SIC cotizan de forma esporádica y traen precio vigente pero
apenas un puñado de observaciones históricas. Se valúan con normalidad, pero
quedan fuera de correlaciones y contribución al riesgo, y el tablero **dice
cuáles** en vez de omitirlas en silencio.

---

## Efectivo

El saldo parte de un **efectivo inicial configurable** (cero por omisión) y
acumula los flujos netos de cada operación. Con el saldo inicial en cero el
número que se muestra es el flujo neto acumulado del periodo, no el saldo real
del contrato: para que el esquema cuadre hay que capturar en el panel lateral el
efectivo real a la fecha de la posición base.

---

## Estructura

```
app.py                  Interfaz y orquestación
portfolio/
  taxonomy.py           Emisora → ticker, sector, industria, región, divisa
  loader.py             Lectura de posición, bitácora y boletas
  engine.py             Posición, efectivo, resultado realizado, valuación
  market.py             Capa Yahoo Finance con caché
  analytics.py          Riesgo, atribución, concentración, diagnóstico
  viz.py                Plantilla Plotly y constructores de figuras
assets/                 Logotipo y hoja de estilo
data/                   Archivos de posición y boletas
```

---

## Diseño

Doble tema con el morado institucional `#522D6D`: la app sigue la preferencia
del sistema (claro u oscuro) vía `st.context.theme` y puede fijarse manualmente
en el menú ⋮ → *Settings* → *Appearance*. Los tokens de la interfaz, la
plantilla de Plotly y el logotipo cambian juntos; el modo oscuro trabaja sobre
`#14141b` y el claro sobre `#fcfcfb`.

La paleta categórica es de orden fijo y cada modo usa el escalón validado para
su superficie: banda de luminosidad, piso de croma, separación bajo
deuteranopia y protanopia (ΔE ≥ 8.4 adyacente), piso de visión normal
(ΔE ≥ 19.3) y contraste ≥ 3:1 en las ocho ranuras.

Los colores de resultado se eligieron por contraste medido, no por apariencia:
`#22c55e` y `#ef4444` para cifras —ambos por encima de 4.5:1— mientras que los
tonos de estado `#0ca30c` y `#d03b3b` quedan para rellenos, donde el umbral
aplicable es 3:1. El morado institucional queda reservado para cromo: a 1.71:1
nunca porta texto. La dirección se marca además con `▲`/`▼`, de modo que el
signo no dependa sólo del color.

---

## Hoja de ruta

La pestaña *Diagnóstico* lista las diez mejoras propuestas en orden de impacto.
Las tres de mayor retorno:

1. **TWR y MWR** — separar la habilidad del gestor del efecto del *timing* de
   las aportaciones. Es lo que exige GIPS para presentar cifras a un cliente
   institucional.
2. **Atribución Brinson-Fachler contra el IPC** — descomponer el resultado en
   asignación, selección e interacción, para poder decir si el valor viene de la
   visión sectorial o de la selección de emisoras.
3. **Persistencia histórica** — guardar la foto diaria de la posición. Hoy el
   histórico se aproxima aplicando la tenencia actual sobre precios pasados, lo
   que no captura el efecto de las operaciones intermedias.

---

## Advertencia

Los precios provienen de Yahoo Finance y pueden presentar retraso de hasta 15
minutos. Documento de trabajo para uso interno; no constituye una recomendación
de inversión.
