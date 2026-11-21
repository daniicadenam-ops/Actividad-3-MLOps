
"""
Tests de monitoreo y simulación de drift para Credit Default.

No necesitan servicios externos, MLflow ni Prefect.
"""

import numpy as np
import pandas as pd
import pytest

from domain.monitoreo import (
    NUMERICAS,
    VARIABLES,
    OBJETIVO,
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    validar_lote,
)


# ============================================================
# 1. FIXTURE: DATOS SIMULADOS DE CREDIT DEFAULT
# ============================================================

@pytest.fixture
def datos():
    """
    Genera un conjunto de datos de prueba con la estructura
    de Credit Default.
    """

    rng = np.random.default_rng(0)

    df = pd.DataFrame({
        columna: rng.normal(0, 1, 500)
        for columna in NUMERICAS
    })

    # Variables categóricas
    df["SEX"] = rng.choice([1, 2], 500)
    df["EDUCATION"] = rng.choice([1, 2, 3, 4], 500)
    df["MARRIAGE"] = rng.choice([1, 2, 3], 500)

    # Variable objetivo
    df[OBJETIVO] = rng.integers(0, 2, 500)

    return df


# ============================================================
# 2. TEST DE SIMULACIÓN DE DRIFT
# ============================================================

def test_drift_cero_no_cambia_los_datos(datos):

    lote = aplicar_drift(datos, 0.0)

    assert lote.equals(datos)


def test_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(datos):

    tasa = datos[OBJETIVO].mean()

    lote = aplicar_drift(datos, 0.6)

    tasa_esperada = tasa + (
        0.6 * SUBIDA_TASA_MAXIMA
    )

    assert lote[OBJETIVO].mean() == pytest.approx(
        tasa_esperada,
        abs=0.01,
    )


def test_drift_cambia_la_fraccion_indicada_de_variables(datos):

    lote = aplicar_drift(datos, 0.6)

    cambiadas = [
        columna
        for columna in NUMERICAS
        if not np.allclose(
            lote[columna],
            datos[columna],
        )
    ]

    assert len(cambiadas) == round(
        0.6 * len(NUMERICAS)
    )


def test_drift_no_modifica_variables_categoricas(datos):

    lote = aplicar_drift(datos, 0.6)

    categoricas = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
    ]

    for columna in categoricas:

        assert lote[columna].equals(
            datos[columna]
        )


def test_drift_rechaza_nivel_fuera_de_rango(datos):

    with pytest.raises(ValueError):

        aplicar_drift(datos, 1.5)


# ============================================================
# 3. TEST DE RESULTADOS DE EVIDENTLY
# ============================================================

def test_fraccion_desde_evidently():

    resultado = {
        "metrics": [
            {
                "metric_name": "ValueDrift(column=AGE)",
                "value": 0.5,
            },
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {
                    "count": 7,
                    "share": 0.5,
                },
            },
        ]
    }

    assert fraccion_desde_evidently(resultado) == 0.5


def test_fraccion_desde_evidently_sin_resultado():

    resultado = {
        "metrics": []
    }

    with pytest.raises(ValueError):

        fraccion_desde_evidently(resultado)


# ============================================================
# 4. TEST DE VALIDACIÓN DEL LOTE
# ============================================================

def test_validar_lote_exige_objetivo(datos):

    lote = datos.drop(
        columns=[OBJETIVO]
    )

    with pytest.raises(
        ValueError,
        match=OBJETIVO,
    ):

        validar_lote(lote)


def test_validar_lote_rechaza_nulos_en_objetivo(datos):

    lote = datos.copy()

    lote.loc[0, OBJETIVO] = np.nan

    with pytest.raises(
        ValueError,
        match=OBJETIVO,
    ):

        validar_lote(lote)


def test_validar_lote_acepta_objetivo_binario(datos):

    validar_lote(datos)


# ============================================================
# 5. TEST DE TARGET DRIFT
# ============================================================

def test_drift_de_target_detecta_cambio_de_tasa():

    referencia = pd.Series(
        [0] * 800 + [1] * 200
    )

    lote = pd.Series(
        [0] * 400 + [1] * 600
    )

    resultado = drift_de_target(
        referencia,
        lote,
        alfa=0.05,
    )

    assert resultado["drift"] is True

    assert resultado["tasa_referencia"] == 0.20

    assert resultado["tasa_lote"] == 0.60


def test_drift_de_target_no_alerta_con_la_misma_tasa():

    referencia = pd.Series(
        [0] * 800 + [1] * 200
    )

    lote = pd.Series(
        [0] * 80 + [1] * 20
    )

    resultado = drift_de_target(
        referencia,
        lote,
        alfa=0.05,
    )

    assert resultado["drift"] is False


def test_drift_de_target_retorna_p_valor_valido():

    referencia = pd.Series(
        [0] * 800 + [1] * 200
    )

    lote = pd.Series(
        [0] * 400 + [1] * 600
    )

    resultado = drift_de_target(
        referencia,
        lote,
        alfa=0.05,
    )

    assert 0 <= resultado["p_valor"] <= 1
