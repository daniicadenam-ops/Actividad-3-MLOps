
"""Tests de datos: esquema, nulos y variable objetivo de Credit Default."""

import pytest

from scripts.load_data import cargar_datos

from domain.monitoreo import (
    OBJETIVO,
    VARIABLES,
)


@pytest.fixture(scope="module")
def df():
    """Carga el dataset de Credit Default una sola vez."""
    return cargar_datos()


def test_tiene_todas_las_columnas_esperadas(df):
    """
    Verifica que el dataset contenga las columnas
    necesarias para el proyecto.
    """

    columnas_esperadas = [
        "ID",
        *VARIABLES,
        OBJETIVO,
    ]

    assert all(
        columna in df.columns
        for columna in columnas_esperadas
    )


def test_no_hay_nulos(df):
    """Verifica que el dataset no tenga valores nulos."""

    assert int(df.isna().sum().sum()) == 0


def test_objetivo_es_binario(df):
    """
    Verifica que la variable objetivo
    contenga únicamente valores 0 y 1.
    """

    assert set(df[OBJETIVO].unique()) <= {0, 1}


def test_id_no_se_utiliza_como_predictor():
    """
    Verifica que ID no forme parte de las
    variables predictoras del modelo.
    """

    assert "ID" not in VARIABLES


def test_variables_predictoras_sin_objetivo():
    """
    Verifica que la variable objetivo no esté
    incluida dentro de las variables predictoras.
    """

    assert OBJETIVO not in VARIABLES
