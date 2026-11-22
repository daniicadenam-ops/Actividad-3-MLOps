
import pandas as pd
import numpy as np
import pytest

from pathlib import Path

from infraestructure.flows import inferencia_flow as flujo


# --------------------------------------------------
# MODELO SIMULADO
# --------------------------------------------------

class ModeloFalso:
    """Simula el comportamiento del modelo registrado en MLflow."""

    feature_names_in_ = np.array([
        "LIMIT_BAL",
        "AGE",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ])

    def predict_proba(self, X):
        assert list(X.columns) == list(self.feature_names_in_)

        return np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
        ])


# --------------------------------------------------
# CARGA DEL MODELO CHAMPION
# --------------------------------------------------

def test_cargar_champion(monkeypatch):
    """Verifica que se cargue el modelo champion y su trazabilidad."""

    class VersionFalsa:
        version = "3"
        run_id = "run_entrenamiento_123"

    class ClienteFalso:
        def get_model_version_by_alias(self, nombre, alias):
            assert nombre == "credit-default-model"
            assert alias == "champion"
            return VersionFalsa()

    monkeypatch.setattr(
        flujo,
        "MlflowClient",
        ClienteFalso,
    )

    monkeypatch.setattr(
        flujo.mlflow.sklearn,
        "load_model",
        lambda uri: ModeloFalso()
        if uri == "models:/credit-default-model@champion"
        else None,
    )

    cfg = {
        "registro": {
            "nombre_modelo": "credit-default-model",
        }
    }

    modelo, version, run_id = flujo.cargar_champion.fn(cfg)

    assert isinstance(modelo, ModeloFalso)
    assert version == "3"
    assert run_id == "run_entrenamiento_123"


# --------------------------------------------------
# INGESTA DE DATOS
# --------------------------------------------------

def test_ingesta_carga_lote(monkeypatch):
    """Verifica que la ingesta genere un lote del tamaño solicitado."""

    datos = pd.DataFrame({
        "ID": range(1, 101),
        "LIMIT_BAL": [30000] * 100,
        "AGE": [30] * 100,
        "SEX": [1] * 100,
        "EDUCATION": [2] * 100,
        "MARRIAGE": [1] * 100,
        "default_payment_next_month": [0] * 100,
    })

    monkeypatch.setattr(
        flujo,
        "leer_datos",
        lambda ruta: datos.copy(),
    )

    cfg = {
        "datos": {
            "ruta": "data/dataset.csv",
        },
        "inferencia": {
            "tamano_lote": 10,
        },
    }

    lote = flujo.ingesta.fn(cfg)

    assert len(lote) == 10
    assert "ID" in lote.columns
    assert "default_payment_next_month" in lote.columns


def test_ingesta_no_supera_datos_disponibles(monkeypatch):
    """Verifica que la ingesta no solicite más registros de los disponibles."""

    datos = pd.DataFrame({
        "ID": [1, 2, 3],
        "AGE": [25, 30, 35],
    })

    monkeypatch.setattr(
        flujo,
        "leer_datos",
        lambda ruta: datos.copy(),
    )

    cfg = {
        "datos": {
            "ruta": "data/dataset.csv",
        },
        "inferencia": {
            "tamano_lote": 100,
        },
    }

    lote = flujo.ingesta.fn(cfg)

    assert len(lote) == 3


# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------

def crear_lote():
    """Construye un lote de prueba con las variables del modelo."""

    return pd.DataFrame({
        "ID": [1, 2, 3],
        "LIMIT_BAL": [30000, 50000, 70000],
        "AGE": [25, 35, 45],
        "SEX": [1, 2, 1],
        "EDUCATION": [2, 1, 2],
        "MARRIAGE": [1, 2, 1],
        "default_payment_next_month": [0, 1, 0],
    })


def test_predecir_genera_probabilidades_y_predicciones(tmp_path):
    """Verifica que se generen probabilidades y clases válidas."""

    modelo = ModeloFalso()
    lote = crear_lote()

    cfg = {
        "registro": {
            "nombre_modelo": "credit-default-model",
        },
        "metricas": {
            "umbral": 0.5,
        },
        "inferencia": {
            "carpeta_salida": str(tmp_path),
        },
    }

    ruta, filas, proba_media = flujo.predecir.fn(
        modelo,
        lote,
        cfg,
    )

    resultado = pd.read_csv(ruta)

    assert Path(ruta).exists()
    assert filas == 3

    assert "probabilidad_default" in resultado.columns
    assert "prediccion_default" in resultado.columns

    assert resultado["probabilidad_default"].between(0, 1).all()

    assert set(
        resultado["prediccion_default"].unique()
    ).issubset({0, 1})

    assert np.isclose(
        proba_media,
        np.mean([0.2, 0.7, 0.4]),
    )


def test_predecir_conserva_columnas_originales(tmp_path):
    """Verifica que la inferencia conserve las columnas del lote."""

    modelo = ModeloFalso()
    lote = crear_lote()

    cfg = {
        "registro": {
            "nombre_modelo": "credit-default-model",
        },
        "metricas": {
            "umbral": 0.5,
        },
        "inferencia": {
            "carpeta_salida": str(tmp_path),
        },
    }

    ruta, _, _ = flujo.predecir.fn(
        modelo,
        lote,
        cfg,
    )

    resultado = pd.read_csv(ruta)

    for columna in lote.columns:
        assert columna in resultado.columns


def test_predecir_clasifica_segun_umbral(tmp_path):
    """Verifica que la clasificación respete el umbral configurado."""

    modelo = ModeloFalso()
    lote = crear_lote()

    cfg = {
        "registro": {
            "nombre_modelo": "credit-default-model",
        },
        "metricas": {
            "umbral": 0.5,
        },
        "inferencia": {
            "carpeta_salida": str(tmp_path),
        },
    }

    ruta, _, _ = flujo.predecir.fn(
        modelo,
        lote,
        cfg,
    )

    resultado = pd.read_csv(ruta)

    assert resultado["prediccion_default"].tolist() == [
        0, 1, 0
    ]


# --------------------------------------------------
# REGISTRO EN MLFLOW
# --------------------------------------------------

def test_registrar_inferencia(monkeypatch, tmp_path):
    """Verifica que se registre la trazabilidad y las métricas."""

    llamadas = {
        "tags": {},
        "metricas": {},
        "artefactos": [],
    }

    class RunFalso:
        info = type(
            "Info",
            (),
            {"run_id": "run_inferencia_123"},
        )()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        flujo.mlflow,
        "start_run",
        lambda **kwargs: RunFalso(),
    )

    monkeypatch.setattr(
        flujo.mlflow,
        "set_tag",
        lambda clave, valor: llamadas["tags"].update({
            clave: valor
        }),
    )

    monkeypatch.setattr(
        flujo.mlflow,
        "log_metric",
        lambda clave, valor: llamadas["metricas"].update({
            clave: valor
        }),
    )

    monkeypatch.setattr(
        flujo.mlflow,
        "log_artifact",
        lambda ruta: llamadas["artefactos"].append(ruta),
    )

    archivo = tmp_path / "predicciones.csv"
    archivo.write_text("ID,prediccion_default\n1,1")

    cfg = {
        "registro": {
            "nombre_modelo": "credit-default-model",
        },
    }

    flujo.registrar_inferencia.fn(
        "3",
        "run_entrenamiento_123",
        str(archivo),
        100,
        0.35,
        cfg,
    )

    assert llamadas["tags"]["fase"] == "inferencia"
    assert llamadas["tags"]["modelo_version"] == "3"
    assert llamadas["tags"]["modelo_run_id"] == "run_entrenamiento_123"

    assert llamadas["metricas"]["filas_predichas"] == 100

    assert llamadas["metricas"]["probabilidad_default_media"] == 0.35

    assert str(archivo) in llamadas["artefactos"]
