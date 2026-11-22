
"""
Actividad 3: monitoreo de drift de features.

Compara las variables de entrada del lote de inferencia con
los datos de entrenamiento usando Evidently.
Genera un reporte HTML y emite alertas de drift.
"""

from pathlib import Path
from datetime import date

import yaml
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from domain.modelo import VARIABLES
from domain.monitoreo import dividir, fraccion_desde_evidently
from scripts.load_data import cargar_datos, cargar_lote


CONFIG = "configs/config.yaml"


@task
def referencia(cfg):
    """Obtiene las variables de los datos de entrenamiento."""

    print("Cargando datos de referencia...")

    entrenamiento, _ = dividir(
        cargar_datos(cfg["datos"]["ruta"]),
        cfg["division"],
    )

    datos_referencia = entrenamiento[VARIABLES].copy()

    print(
        f"Referencia cargada: {len(datos_referencia)} registros "
        f"y {len(VARIABLES)} variables."
    )

    return datos_referencia


@task
def calcular_drift(datos_referencia, ruta_lote, cfg):
    """Compara el lote actual con la referencia y genera el reporte."""

    print("\nIniciando evaluación de Drift con Evidently...")

    ruta_lote = Path(ruta_lote)

    if not ruta_lote.exists():
        raise FileNotFoundError(
            f"No existe el archivo de inferencia: {ruta_lote}"
        )

    lote = cargar_lote(str(ruta_lote))

    datos_actuales = lote[VARIABLES].copy()

    print(f"Lote evaluado: {ruta_lote}")
    print(f"Registros actuales: {len(datos_actuales)}")
    print(f"Variables evaluadas: {len(VARIABLES)}")

    reporte = Report([
        DataDriftPreset()
    ])

    resultado = reporte.run(
        reference_data=datos_referencia,
        current_data=datos_actuales,
    )

    # Crear carpeta de reportes
    carpeta = Path(
        cfg["monitoreo"]["carpeta_reportes"]
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Nombre independiente para cada escenario
    ruta_html = carpeta / (
        f"drift_features_{ruta_lote.stem}.html"
    )

    # Guardar reporte HTML
    resultado.save_html(str(ruta_html))

    if not ruta_html.exists():
        raise RuntimeError(
            "Evidently no generó el archivo HTML."
        )

    print("\nREPORTE GENERADO CORRECTAMENTE")
    print(f"Ubicación: {ruta_html.resolve()}")

    # Obtener resultados estructurados de Evidently
    resultado_dict = resultado.dict()

    fraccion = fraccion_desde_evidently(
        resultado_dict
    )

    print(
        f"Proporción de variables con Drift: {fraccion:.2%}"
    )

    return fraccion


@task
def alerta(fraccion, umbral):
    """Genera una alerta cuando el drift supera el umbral."""

    hay_alerta = fraccion > umbral

    print("\n--- RESULTADO DEL MONITOREO ---")

    if hay_alerta:
        print(
            f"ALERTA: Drift de Features en {fraccion:.2%}. "
            f"Umbral configurado: {umbral:.2%}"
        )
    else:
        print(
            f"Sin alerta: Drift de Features en {fraccion:.2%}. "
            f"Umbral configurado: {umbral:.2%}"
        )

    return hay_alerta


@flow(
    name="drift-features",
    log_prints=True,
)
def drift_features_flow():

    # 1. Leer configuración
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Obtener nivel de Drift
    nivel_drift = float(
        cfg["inferencia"]["drift"]["nivel"]
    )

    if nivel_drift not in (0.0, 0.3, 0.6):
        raise ValueError(
            "El nivel de Drift debe ser 0.0, 0.3 o 0.6"
        )

    # 3. Construir ruta del archivo de inferencia
    nivel_nombre = str(nivel_drift).replace(".", "_")

    nombre_archivo = (
        f"predicciones_credit_default_"
        f"drift_{nivel_nombre}_"
        f"{date.today().isoformat()}.csv"
    )

    ruta_lote = (
        Path(cfg["inferencia"]["carpeta_salida"])
        / nombre_archivo
    )

    print("\n===================================")
    print("MONITOREO DE DRIFT DE FEATURES")
    print("===================================")
    print(f"Nivel de Drift: {nivel_drift}")
    print(f"Archivo seleccionado: {ruta_lote.resolve()}")

    # 4. Validar que exista el lote
    if not ruta_lote.exists():
        raise FileNotFoundError(
            f"No existe el lote: {ruta_lote.resolve()}. "
            "Ejecuta primero inferencia_flow.py."
        )

    # 5. Cargar referencia
    datos_referencia = referencia(cfg)

    # 6. Calcular Drift y generar HTML
    fraccion = calcular_drift(
        datos_referencia,
        str(ruta_lote),
        cfg,
    )

    # 7. Evaluar umbral y generar alerta
    resultado_alerta = alerta(
        fraccion,
        float(cfg["monitoreo"]["umbral_drift"]),
    )

    print("\nMonitoreo de Features finalizado.")
    print(f"Reporte guardado en: {Path(cfg['monitoreo']['carpeta_reportes']).resolve()}")

    return resultado_alerta


if __name__ == "__main__":
    drift_features_flow()
