
"""
Tests del modelo Credit Default:
features, pipeline y métricas.

No necesitan MLflow ni Prefect.
"""

import numpy as np
import pandas as pd
import pytest

from domain.modelo import (
    NUMERICAS,
    CATEGORICAS,
    calcular_metricas,
    construir_modelo,
    resolver_features,
)


# ============================================================
# 1. TEST DE RESOLUCIÓN DE FEATURES
# ============================================================

def test_resolver_features_all():

    numericas, categoricas = resolver_features("ALL")

    assert numericas == NUMERICAS
    assert categoricas == CATEGORICAS


def test_resolver_features_lista_separa_numericas_y_categoricas():

    numericas, categoricas = resolver_features(
        ["AGE", "SEX"]
    )

    assert numericas == ["AGE"]
    assert categoricas == ["SEX"]


def test_resolver_features_rechaza_variables_desconocidas():

    with pytest.raises(ValueError):
        resolver_features(["VARIABLE_INEXISTENTE"])


# ============================================================
# 2. TEST DE CONSTRUCCIÓN Y PREDICCIÓN DEL MODELO
# ============================================================

@pytest.mark.parametrize(
    "tipo, parametros",
    [
        (
            "logistic_regression",
            {
                "C": 1.0,
                "max_iter": 1000,
            },
        ),
        (
            "random_forest",
            {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42,
            },
        ),
    ],
)
def test_modelo_predice_probabilidades_validas(tipo, parametros):

    rng = np.random.default_rng(42)

    # Simulación de datos con estructura Credit Default
    X = pd.DataFrame({
        "LIMIT_BAL": rng.integers(10000, 200000, 200),
        "AGE": rng.integers(21, 70, 200),
        "SEX": rng.choice([1, 2], 200),
    })

    # Variable objetivo binaria simulada
    y = (
        (X["AGE"] > 45) |
        (X["LIMIT_BAL"] < 50000)
    ).astype(int)

    # Construcción del pipeline
    modelo = construir_modelo(
        tipo,
        parametros,
        ["LIMIT_BAL", "AGE"],
        ["SEX"],
    )

    # Entrenamiento
    modelo.fit(X, y)

    # Predicción de probabilidades
    proba = modelo.predict_proba(X)[:, 1]

    assert proba.shape == (200,)

    assert np.isfinite(proba).all()

    assert ((proba >= 0) & (proba <= 1)).all()


# ============================================================
# 3. TEST DE MÉTRICAS
# ============================================================

def test_calcular_metricas_perfecto():

    y = np.array([0, 0, 1, 1])

    proba = np.array([0.1, 0.2, 0.8, 0.9])

    metricas = calcular_metricas(
        y,
        proba,
        umbral=0.5,
    )

    assert metricas["auc"] == 1.0
    assert metricas["accuracy"] == 1.0


def test_calcular_metricas_retorna_valores_validos():

    y = np.array([0, 0, 1, 1, 0, 1])

    proba = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.9])

    metricas = calcular_metricas(
        y,
        proba,
        umbral=0.5,
    )

    assert 0 <= metricas["auc"] <= 1
    assert 0 <= metricas["accuracy"] <= 1
