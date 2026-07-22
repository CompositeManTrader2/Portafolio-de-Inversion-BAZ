"""
Lectura de los insumos del portafolio.

Dos fuentes:
  1. Posicion base   -> hoja 'Hoja1' del archivo de Posicion Capitales.
                        Es la foto de la cartera al cierre del miercoles previo,
                        ANTES de aplicar los movimientos subsecuentes.
  2. Movimientos     -> hoja 'Coste' del mismo archivo, o bien las boletas
                        del custodio (formato Res.NNNNNN), que traen ademas
                        comision e IVA reales.

Los montos de 'Hoja1' vienen expresados en millones y con redondeos; se
recalculan siempre como titulos x precio para mantener consistencia interna.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from .taxonomy import clasificar, extraer_tipo_valor, normalizar_emisora

_WS = re.compile(r"\s+")

# Fecha de la foto base: cierre del miercoles previo al primer movimiento.
FECHA_BASE_DEFAULT = date(2026, 7, 15)


class ErrorDeCarga(ValueError):
    """El archivo no tiene la forma esperada."""


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _a_fecha(v) -> date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.to_datetime(v, dayfirst=True).date()
    except Exception:
        return None


def _a_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return None if pd.isna(v) else float(v)
    s = re.sub(r"[^0-9eE\.\-+]", "", str(v).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def _abrir(origen) -> pd.ExcelFile:
    """Acepta ruta, bytes o file-like de Streamlit."""
    if isinstance(origen, (str, Path)):
        return pd.ExcelFile(origen, engine="openpyxl")
    if isinstance(origen, bytes):
        return pd.ExcelFile(BytesIO(origen), engine="openpyxl")
    return pd.ExcelFile(origen, engine="openpyxl")


def _sin_acentos(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def _clave(s) -> str:
    """Normaliza un nombre de columna para poder compararlo."""
    return _WS.sub(" ", _sin_acentos(s).strip().lower()).replace(" /", "/")


def _localizar_encabezado(df: pd.DataFrame, requeridas: set[str]) -> int | None:
    """Encuentra la fila que funge como encabezado buscando las columnas dadas."""
    for i in range(min(len(df), 30)):
        celdas = {
            _clave(c)
            for c in df.iloc[i].tolist()
            if c is not None and not pd.isna(c)
        }
        if all(any(req in c for c in celdas) for req in requeridas):
            return i
    return None


def _es_continuacion(fila: pd.Series) -> bool:
    """
    True si la fila parece la segunda linea de un encabezado partido en dos
    (solo texto, sin numeros ni fechas). Las boletas del custodio parten el
    encabezado en 'Fecha de' / 'Operacion'.
    """
    valores = [v for v in fila.tolist() if v is not None and not pd.isna(v)]
    if not 1 <= len(valores) <= 8:
        return False
    return all(isinstance(v, str) and not _a_float(v) for v in valores)


def _leer_con_encabezado(xls, hoja: str, fila: int,
                         crudo: pd.DataFrame) -> pd.DataFrame:
    """
    Lee la hoja usando `fila` como encabezado, fusionando la fila siguiente
    cuando el encabezado viene partido en dos renglones.
    """
    siguiente = fila + 1
    fusionar = siguiente < len(crudo) and _es_continuacion(crudo.iloc[siguiente])

    if not fusionar:
        df = pd.read_excel(xls, sheet_name=hoja, header=fila)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    superior = crudo.iloc[fila].tolist()
    inferior = crudo.iloc[siguiente].tolist()
    nombres = []
    for arriba, abajo in zip(superior, inferior):
        partes = [str(x).strip() for x in (arriba, abajo)
                  if x is not None and not pd.isna(x)]
        nombres.append(" ".join(partes) if partes else "")

    df = pd.read_excel(xls, sheet_name=hoja, header=None, skiprows=siguiente + 1)
    df = df.iloc[:, :len(nombres)]
    df.columns = nombres
    return df


# --------------------------------------------------------------------------
# Posicion base
# --------------------------------------------------------------------------

def _buscador_de_columnas(df: pd.DataFrame):
    """
    Devuelve una funcion que resuelve el nombre real de una columna a partir
    de una lista de alias, primero por coincidencia exacta y luego por
    prefijo/subcadena. Tolera encabezados fusionados como
    'Fecha de Operacion' o 'Compra /Venta'.
    """
    mapa = {_clave(c): c for c in df.columns}

    def col(*alias: str) -> str | None:
        for a in alias:
            if a in mapa:
                return mapa[a]
        for a in alias:
            for k, original in mapa.items():
                if k.startswith(a) or a in k:
                    return original
        return None

    return col


@dataclass(frozen=True)
class PosicionBase:
    df: pd.DataFrame
    fecha: date
    hoja: str


def leer_posicion_base(origen, hoja: str | None = None,
                       fecha: date | None = None) -> PosicionBase:
    """
    Devuelve un DataFrame con: emisora, tipo_valor, titulos, precio_costo,
    costo_total, precio_ref, mas las columnas taxonomicas.
    """
    xls = _abrir(origen)
    hoja = hoja or xls.sheet_names[0]
    if hoja not in xls.sheet_names:
        raise ErrorDeCarga(f"La hoja '{hoja}' no existe. Hojas: {xls.sheet_names}")

    crudo = pd.read_excel(xls, sheet_name=hoja, header=None)
    fila = _localizar_encabezado(crudo, {"emisora", "titulos"})
    if fila is None:
        fila = _localizar_encabezado(crudo, {"emisora"})
    if fila is None:
        raise ErrorDeCarga(
            f"No se encontro un encabezado con 'Emisora' en la hoja '{hoja}'."
        )

    df = _leer_con_encabezado(xls, hoja, fila, crudo)
    col = _buscador_de_columnas(df)

    c_emi = col("emisora")
    c_tit = col("titulos")
    c_pc = col("precio compra", "precio costo", "costo")
    c_pm = col("precio merc", "precio mercado", "precio")
    if not c_emi or not c_tit:
        raise ErrorDeCarga("Faltan las columnas 'Emisora' y/o 'Titulos'.")

    filas = []
    for _, r in df.iterrows():
        cruda = r.get(c_emi)
        if cruda is None or pd.isna(cruda) or not str(cruda).strip():
            continue
        titulos = _a_float(r.get(c_tit))
        precio_costo = _a_float(r.get(c_pc)) if c_pc else None
        if titulos is None or titulos == 0 or precio_costo is None:
            continue
        emisora = normalizar_emisora(cruda)
        tax = clasificar(emisora)
        filas.append({
            "emisora": emisora,
            "tipo_valor": extraer_tipo_valor(cruda),
            "titulos": titulos,
            "precio_costo": precio_costo,
            # Se recalcula: la hoja trae millones y redondeos.
            "costo_total": titulos * precio_costo,
            "precio_ref": _a_float(r.get(c_pm)) if c_pm else None,
            **{k: tax[k] for k in
               ("ticker", "clase_activo", "sector", "industria",
                "region", "mercado", "divisa_subyacente")},
        })

    if not filas:
        raise ErrorDeCarga(f"La hoja '{hoja}' no contiene posiciones legibles.")

    return PosicionBase(
        df=pd.DataFrame(filas).reset_index(drop=True),
        fecha=fecha or FECHA_BASE_DEFAULT,
        hoja=hoja,
    )


# --------------------------------------------------------------------------
# Movimientos
# --------------------------------------------------------------------------

COLUMNAS_MOV = ["fecha", "operacion", "emisora", "titulos", "precio",
                "comision", "iva", "importe_neto", "fuente"]


def _mov_vacio() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNAS_MOV)


def _normalizar_operacion(v) -> str | None:
    """'C'/'Compra' -> 'C'; 'V'/'Venta' -> 'V'."""
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s.startswith("C"):
        return "C"
    if s.startswith("V"):
        return "V"
    return None


def leer_movimientos(origen, hoja: str | None = None,
                     fuente: str = "hoja Coste") -> pd.DataFrame:
    """
    Lee una bitacora de operaciones con columnas
    Fecha | Operacion | Emisora | Titulos | Precio  (+ opcional Comision / IVA).
    """
    xls = _abrir(origen)
    if hoja is None:
        for cand in ("Coste", "Movimientos", "Operaciones"):
            if cand in xls.sheet_names:
                hoja = cand
                break
        else:
            hoja = xls.sheet_names[-1]
    if hoja not in xls.sheet_names:
        return _mov_vacio()

    crudo = pd.read_excel(xls, sheet_name=hoja, header=None)
    fila = _localizar_encabezado(crudo, {"emisora", "precio"})
    if fila is None:
        return _mov_vacio()

    df = _leer_con_encabezado(xls, hoja, fila, crudo)
    col = _buscador_de_columnas(df)

    # 'fecha de operacion' antes que 'fecha de liquidacion': la posicion se
    # reconoce en la fecha de concertacion, no en la de liquidacion.
    c_f = col("fecha de operacion", "fecha operacion", "fecha")
    c_o = col("compra/venta", "operacion", "compra")
    c_e = col("emisora", "emisora serie")
    c_t = col("titulos", "cantidad")
    c_p = col("precio")
    c_com = col("comision")
    c_iva = col("i.v.a.", "iva")
    c_neto = col("monto neto", "importe neto")

    if not all((c_o, c_e, c_t, c_p)):
        return _mov_vacio()

    filas = []
    for _, r in df.iterrows():
        op = _normalizar_operacion(r.get(c_o))
        emi = r.get(c_e)
        titulos = _a_float(r.get(c_t))
        precio = _a_float(r.get(c_p))
        if op is None or emi is None or pd.isna(emi):
            continue
        if not titulos or precio is None:
            continue
        filas.append({
            "fecha": _a_fecha(r.get(c_f)) if c_f else None,
            "operacion": op,
            "emisora": normalizar_emisora(emi),
            "titulos": abs(titulos),
            "precio": precio,
            "comision": _a_float(r.get(c_com)) if c_com else None,
            "iva": _a_float(r.get(c_iva)) if c_iva else None,
            "importe_neto": _a_float(r.get(c_neto)) if c_neto else None,
            "fuente": fuente,
        })

    if not filas:
        return _mov_vacio()
    return pd.DataFrame(filas)[COLUMNAS_MOV]


def leer_operaciones(origen, fuente: str | None = None) -> pd.DataFrame:
    """
    Lee un archivo de operaciones de cualquier procedencia: una boleta del
    custodio (formato 'Res.NNNNNN'), un reporte de la mesa o una bitacora
    capturada a mano.

    Se recorren TODAS las hojas y se acumula lo que cada una aporte. Limitarse
    a la primera hoja hacia que un archivo con portada o indice delante de los
    datos devolviera cero operaciones sin avisar, que es peor que fallar: el
    usuario cree que cargo sus movimientos y no cargo nada.
    """
    xls = _abrir(origen)
    etiqueta = fuente or "archivo de operaciones"

    bloques = []
    for hoja in xls.sheet_names:
        try:
            parcial = leer_movimientos(origen, hoja=hoja, fuente=etiqueta)
        except Exception:
            continue
        if len(parcial):
            bloques.append(parcial)

    if not bloques:
        return _mov_vacio()
    return pd.concat(bloques, ignore_index=True)


# Nombre previo, conservado para no romper llamadas existentes.
leer_boleta_custodio = leer_operaciones


def consolidar_movimientos(*bloques: pd.DataFrame) -> pd.DataFrame:
    """
    Une varias fuentes de movimientos y elimina duplicados.

    Una operacion se considera repetida si coincide en
    (fecha, operacion, emisora, titulos, precio). Esto permite cargar la
    boleta del custodio encima de la bitacora interna sin doble conteo:
    se conserva la primera aparicion, que es la de mayor prioridad segun
    el orden en que se pasen los bloques.
    """
    vivos = [b for b in bloques if b is not None and len(b)]
    if not vivos:
        return _mov_vacio()

    df = pd.concat(vivos, ignore_index=True)

    # Una operacion se identifica por emisora, sentido, titulos y precio, y
    # dos registros con esa misma llave se fusionan SOLO si sus fechas son
    # compatibles: iguales, o una de las dos vacia. La bitacora interna y la
    # boleta pueden diferir en fecha (concertacion vs liquidacion) o traerla
    # vacia, y ahi la fusion es correcta; pero dos operaciones legitimas con
    # los mismos parametros en dias distintos (recomprar el mismo lote al
    # mismo precio dias despues) deben sobrevivir como operaciones separadas.
    # Al fusionar se conserva el primer valor no nulo de cada campo, de modo
    # que la boleta aporta comision e IVA reales y la bitacora la fecha.
    consolidadas: list[dict] = []
    indice: dict[tuple, list[int]] = {}

    for _, fila in df.iterrows():
        llave = (fila["operacion"], fila["emisora"],
                 float(fila["titulos"]), round(float(fila["precio"]), 6))
        fecha = fila["fecha"] if not pd.isna(fila["fecha"]) else None

        destino = None
        for i in indice.get(llave, []):
            fecha_prev = consolidadas[i]["fecha"]
            if fecha is None or fecha_prev is None or fecha == fecha_prev:
                destino = i
                break

        if destino is None:
            nueva = {c: (None if pd.isna(fila[c]) else fila[c])
                     for c in COLUMNAS_MOV}
            consolidadas.append(nueva)
            indice.setdefault(llave, []).append(len(consolidadas) - 1)
        else:
            entrada = consolidadas[destino]
            for c in ("fecha", "comision", "iva", "importe_neto"):
                if entrada[c] is None and not pd.isna(fila[c]):
                    entrada[c] = fila[c]
            fuentes = [f for f in (entrada["fuente"], fila["fuente"]) if f]
            entrada["fuente"] = " + ".join(dict.fromkeys(fuentes))

    df = pd.DataFrame(consolidadas, columns=COLUMNAS_MOV)

    # Orden cronologico estable; las operaciones sin fecha van al final.
    df["_sin_fecha"] = df["fecha"].isna()
    df = df.sort_values(["_sin_fecha", "fecha"], kind="stable")
    return df.drop(columns="_sin_fecha").reset_index(drop=True)
