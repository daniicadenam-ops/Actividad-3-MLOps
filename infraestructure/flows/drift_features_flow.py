
"""
Actividad 3: monitoreo de drift de features.

Compara las variables de entrada del lote de inferencia
con las variables del conjunto de entrenamiento usando Evidently.

Proyecto: Credit Default.

Genera un reporte HTML y emite una alerta cuando la
fracción de variables con drift supera el umbral configurado.

Ejecución desde la raíz del proyecto:
    poetry run python -m infraestructure.flows.drift_features_flow
"""

from pathlib import Path

import pandas as pd
import yaml

from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from domain.monitoreo import (
    VARIABLES,
    dividir,
    fraccion_desde_evidently,
)

from scripts.load_data import cargar_datos


CONFIG = "configs/config.yaml"


@task
def referencia(cfg):
    """
    Obtiene las variables predictoras del conjunto
    de entrenamiento. Se utilizan como referencia.
    """

    df = cargar_datos(cfg["datos"]["ruta"])

    entrenamiento, _ = dividir(
        df,
        cfg["division"],
    )

    datos_referencia = entrenamiento[VARIABLES].copy()

    print(
        f"Datos de referencia cargados: "
        f"{datos_referencia.shape[0]} registros"
    )

    print(
        f"Variables utilizadas para monitoreo: "
        f"{len(VARIABLES)}"
    )

    return datos_referencia


@task
def calcular_drift(datos_referencia, ruta_lote, cfg):
    """
    Ejecuta el análisis de drift con Evidently
    y guarda el reporte HTML.
    """

    ruta = Path(ruta_lote)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el lote de inferencia: {ruta}"
        )

    # El archivo de inferencia se genera como CSV estándar
    lote = pd.read_csv(ruta)

    # Verificar que existan todas las variables predictoras
    faltantes = [
        columna
        for columna in VARIABLES
        if columna not in lote.columns
    ]

    if faltantes:
        raise ValueError(
            f"Al lote de inferencia le faltan variables: {faltantes}"
        )

    # Seleccionar únicamente las variables de entrada
    datos_actuales = lote[VARIABLES].copy()

    print(
        f"Lote actual cargado: "
        f"{datos_actuales.shape[0]} registros"
    )

    # Ejecutar análisis de drift
    resultado = Report(
        [DataDriftPreset()]
    ).run(
        reference_data=datos_referencia,
        current_data=datos_actuales,
    )

    # Crear carpeta para reportes
    carpeta = Path(
        cfg["monitoreo"]["carpeta_reportes"]
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Guardar reporte HTML
    ruta_html = carpeta / (
        f"drift_features_{ruta.stem}.html"
    )

    resultado.save_html(str(ruta_html))

    print(
        f"Reporte de drift de features guardado en: "
        f"{ruta_html}"
    )

    # Obtener fracción de variables con drift
    fraccion = fraccion_desde_evidently(
        resultado.dict()
    )

    print(
        f"Fracción de variables con drift: {fraccion:.2%}"
    )

    return fraccion


@task
def alerta(fraccion, umbral):
    """
    Genera una alerta cuando la proporción de variables
    con drift supera el umbral configurado.
    """

    hay_alerta = fraccion > umbral

    if hay_alerta:

        print(
            f"ALERTA: drift de features en "
            f"{fraccion:.0%} de las variables "
            f"(umbral {umbral:.0%})"
        )

    else:

        print(
            f"Sin alerta: drift de features en "
            f"{fraccion:.0%} de las variables "
            f"(umbral {umbral:.0%})"
        )

    return hay_alerta


@flow(
    name="drift-features-credit-default",
    log_prints=True,
)
def drift_features_flow():

    with open(CONFIG, encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)

    # Obtener referencia del entrenamiento
    datos_referencia = referencia(cfg)

    # Comparar contra el lote de inferencia
    fraccion = calcular_drift(
        datos_referencia,
        cfg["monitoreo"]["lote_actual"],
        cfg,
    )

    # Evaluar si se genera una alerta
    return alerta(
        fraccion,
        cfg["monitoreo"]["umbral_drift"],
    )


if __name__ == "__main__":
    drift_features_flow()
