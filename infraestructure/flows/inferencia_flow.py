
"""
Actividad 3: Pipeline de inferencia Credit Default con emulación de Drift.

Pasos:
cargar modelo champion -> ingesta del lote -> emular Drift
-> predicción -> guardar CSV -> registrar en MLflow.

Niveles de emulación permitidos:
    0.0: Sin Drift.
    0.3: Drift moderado.
    0.6: Drift alto.

El modelo se busca por el alias `champion`.
La inferencia utiliza automáticamente el modelo promovido
por el pipeline de entrenamiento.

Requisitos:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000

Ejecución:
    poetry run python infraestructure/flows/inferencia_flow.py
"""

import os
from datetime import date
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml

from mlflow.tracking import MlflowClient
from prefect import flow, task

from scripts.load_data import cargar_datos as leer_datos
from domain.monitoreo import emular_drift


CONFIG = "configs/config.yaml"


@task
def cargar_champion(cfg):
    """
    Carga el modelo champion desde MLflow.

    Devuelve:
        modelo, version_modelo, run_id_entrenamiento
    """

    print("Buscando modelo champion en MLflow...")

    nombre = cfg["registro"]["nombre_modelo"]

    client = MlflowClient()

    version = client.get_model_version_by_alias(
        nombre,
        "champion",
    )

    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}@champion"
    )

    print(
        f"Modelo cargado: {nombre} | "
        f"Versión={version.version} | "
        f"Run entrenamiento={version.run_id}"
    )

    return modelo, str(version.version), version.run_id


@task
def ingesta(cfg):
    """
    Simula la ingesta de un lote de clientes
    utilizando una muestra del dataset.
    """

    tamano_lote = cfg["inferencia"]["tamano_lote"]

    print(f"Ingestando lote de {tamano_lote} clientes...")

    df = leer_datos(cfg["datos"]["ruta"])

    n = min(tamano_lote, len(df))

    lote = df.sample(
        n=n,
        random_state=42,
    ).copy()

    print(f"Lote cargado: {len(lote)} registros")

    return lote


@task
def predecir(modelo, lote, cfg):
    """
    Aplica la emulación de Drift, calcula las probabilidades
    de incumplimiento y guarda las predicciones en un CSV.
    """

    print("Preparando variables para inferencia...")

    # Obtener nivel configurable de Drift
    nivel_drift = float(
        cfg["inferencia"]["drift"]["nivel"]
    )

    # Validar y aplicar emulación de Drift
    print(f"Aplicando emulación de Drift: {nivel_drift}")

    lote_modificado = emular_drift(
        lote,
        nivel_drift,
    )

    # Utilizar las mismas variables del modelo entrenado
    variables = list(modelo.feature_names_in_)

    X = lote_modificado[variables]

    print(
        f"Prediciendo con modelo "
        f"{cfg['registro']['nombre_modelo']}..."
    )

    # Probabilidad de incumplimiento (clase 1)
    proba = modelo.predict_proba(X)[:, 1]

    resultado = lote_modificado.copy()

    resultado["probabilidad_default"] = proba

    resultado["prediccion_default"] = (
        proba >= cfg["metricas"]["umbral"]
    ).astype(int)

    # Guardar CSV
    print("Guardando predicciones en CSV...")

    carpeta = Path(
        cfg["inferencia"]["carpeta_salida"]
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Diferenciar los archivos según el nivel de Drift
    nivel_nombre = str(nivel_drift).replace(".", "_")

    ruta = carpeta / (
        f"predicciones_credit_default_"
        f"drift_{nivel_nombre}_"
        f"{date.today().isoformat()}.csv"
    )

    resultado.to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )

    print(f"Predicciones guardadas en {ruta}")

    return (
        str(ruta),
        len(proba),
        float(proba.mean()),
    )


@task
def registrar_inferencia(
    version,
    run_id_entrenamiento,
    ruta,
    filas,
    proba_media,
    cfg,
):
    """
    Registra el resultado de inferencia en MLflow
    con trazabilidad del modelo y del nivel de Drift aplicado.
    """

    nivel_drift = float(
        cfg["inferencia"]["drift"]["nivel"]
    )

    print(
        f"Registrando inferencia para modelo "
        f"versión {version}..."
    )

    with mlflow.start_run(
        run_name=(
            f"inferencia-credit-default-"
            f"drift-{nivel_drift}-"
            f"{date.today().isoformat()}"
        )
    ) as run:

        # Tags de trazabilidad
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)

        mlflow.set_tag(
            "modelo_run_id",
            run_id_entrenamiento,
        )

        mlflow.set_tag(
            "modelo_nombre",
            cfg["registro"]["nombre_modelo"],
        )

        mlflow.set_tag(
            "dataset",
            "credit-default",
        )

        # Registrar nivel de Drift aplicado
        mlflow.set_tag(
            "nivel_drift",
            str(nivel_drift),
        )

        # Métricas de inferencia
        mlflow.log_metric(
            "filas_predichas",
            filas,
        )

        mlflow.log_metric(
            "probabilidad_default_media",
            proba_media,
        )

        mlflow.log_metric(
            "nivel_drift",
            nivel_drift,
        )

        # Guardar archivo de predicciones como artefacto
        mlflow.log_artifact(ruta)

        print(
            f"Inferencia registrada en MLflow. "
            f"Run ID={run.info.run_id}"
        )


@flow(
    name="inferencia-credit-default",
    log_prints=True,
)
def inferencia_flow():

    # 1. Leer configuración
    with open(
        CONFIG,
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    # 2. Validar configuración de Drift
    nivel_drift = float(
        cfg["inferencia"]["drift"]["nivel"]
    )

    if nivel_drift not in (0.0, 0.3, 0.6):
        raise ValueError(
            "El nivel de Drift debe ser 0.0, 0.3 o 0.6"
        )

    # 3. Configurar MLflow
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        cfg["mlflow"]["tracking_uri"],
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        cfg["mlflow"]["experimento"]
    )

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Nivel de Drift seleccionado: {nivel_drift}")

    # 4. Cargar modelo champion
    modelo, version, run_id_entrenamiento = cargar_champion(
        cfg
    )

    # 5. Ingestar lote
    lote = ingesta(cfg)

    # 6. Aplicar Drift y generar predicciones
    ruta, filas, proba_media = predecir(
        modelo,
        lote,
        cfg,
    )

    # 7. Registrar inferencia en MLflow
    registrar_inferencia(
        version,
        run_id_entrenamiento,
        ruta,
        filas,
        proba_media,
        cfg,
    )

    print("\nPipeline de inferencia finalizado correctamente.")

    return ruta


if __name__ == "__main__":
    inferencia_flow()
