
"""Tests de la API Credit Default con un modelo falso (sin MLflow)."""

import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest

from fastapi.testclient import TestClient


# ============================================================
# 1. DATOS DE PRUEBA
# ============================================================

CLIENTE = {
    "LIMIT_BAL": 50000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 10000.0,
    "BILL_AMT2": 9000.0,
    "BILL_AMT3": 8000.0,
    "BILL_AMT4": 7000.0,
    "BILL_AMT5": 6000.0,
    "BILL_AMT6": 5000.0,
    "PAY_AMT1": 2000.0,
    "PAY_AMT2": 2000.0,
    "PAY_AMT3": 2000.0,
    "PAY_AMT4": 2000.0,
    "PAY_AMT5": 2000.0,
    "PAY_AMT6": 2000.0,
}


# ============================================================
# 2. MODELO FALSO
# ============================================================

class ModeloFalso:
    """
    Simula el comportamiento del modelo registrado en MLflow.
    Devuelve una probabilidad fija de incumplimiento del 70%.
    """

    feature_names_in_ = np.array(list(CLIENTE.keys()))

    def predict_proba(self, X):
        return np.tile(
            [0.3, 0.7],
            (len(X), 1),
        )


# ============================================================
# 3. FIXTURE DE LA API
# ============================================================

@pytest.fixture
def client(monkeypatch):

    # Simular la carga del modelo desde MLflow
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda uri: ModeloFalso(),
    )

    # Simular la versión del modelo registrada
    monkeypatch.setattr(
        mlflow.tracking,
        "MlflowClient",
        lambda: SimpleNamespace(
            get_model_version_by_alias=lambda nombre, alias:
                SimpleNamespace(version="7")
        ),
    )

    # Recargar el módulo para aplicar los mocks antes
    # de que la API intente cargar el modelo.
    sys.modules.pop(
        "infraestructure.api.main",
        None,
    )

    import infraestructure.api.main as main

    return TestClient(main.app)


# ============================================================
# 4. TEST DE HEALTH Y PREDICCIÓN
# ============================================================

def test_health_y_predict(client):

    # Validar endpoint de salud
    respuesta_health = client.get("/health")

    assert respuesta_health.status_code == 200
    assert respuesta_health.json()["version"] == "7"

    # Validar endpoint de predicción
    respuesta_predict = client.post(
        "/predict",
        json=CLIENTE,
    )

    assert respuesta_predict.status_code == 200

    resultado = respuesta_predict.json()

    assert resultado["probabilidad_default"] == 0.7
    assert resultado["prediccion_default"] == 1
    assert resultado["version_modelo"] == "7"
    assert "umbral" in resultado
