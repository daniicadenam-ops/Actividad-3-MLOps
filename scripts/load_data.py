
"""
Carga y valida el dataset Default of Credit Card Clients
y los lotes generados durante la inferencia.

Por defecto lee datos.ruta de configs/config.yaml.

Ejecución:
    poetry run python scripts/load_data.py

Uso desde los flows:
    from scripts.load_data import cargar_datos, cargar_lote
"""

from pathlib import Path

import pandas as pd
import yaml


RUTA_CONFIG = "configs/config.yaml"

# Columnas originales del dataset.
COLUMNAS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default_payment_next_month",
]

TARGET = "default_payment_next_month"

FEATURES = [
    col for col in COLUMNAS
    if col not in ["ID", TARGET]
]

# Columnas adicionales generadas por la inferencia.
COLUMNAS_INFERENCIA = [
    "probabilidad_default",
    "prediccion_default",
]


def _leer_csv(ruta):
    """
    Lee un CSV, detecta el separador y normaliza
    los nombres de las columnas.
    """

    df = pd.read_csv(
        ruta,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )

    # Eliminar BOM, espacios y normalizar nombres.
    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(" ", "_")
    )

    return df


def _validar_target(df):
    """Valida que el target sea binario cuando está presente."""

    if TARGET in df.columns:
        valores_target = set(df[TARGET].dropna().unique())

        if not valores_target.issubset({0, 1}):
            raise ValueError(
                f"El target contiene valores inesperados: "
                f"{valores_target}"
            )


def cargar_datos(ruta=None):
    """
    Carga el dataset original de entrenamiento.

    Exige las 25 columnas originales.
    """

    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            ruta = yaml.safe_load(f)["datos"]["ruta"]

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta.resolve()}"
        )

    df = _leer_csv(ruta)

    # Validar cantidad de columnas.
    if df.shape[1] != len(COLUMNAS):
        raise ValueError(
            f"El archivo tiene {df.shape[1]} columnas, "
            f"pero se esperaban {len(COLUMNAS)}."
        )

    # Validar nombres de columnas.
    if set(df.columns) != set(COLUMNAS):
        raise ValueError(
            "Las columnas del dataset no coinciden con las esperadas.\n"
            f"Esperadas: {COLUMNAS}\n"
            f"Encontradas: {list(df.columns)}"
        )

    _validar_target(df)

    return df[COLUMNAS].copy()


def cargar_lote(ruta):
    """
    Carga un lote de inferencia.

    Permite columnas adicionales como:
    - probabilidad_default
    - prediccion_default

    Exige las 23 variables predictoras originales.
    No exige que el lote tenga exactamente 25 columnas.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el lote de inferencia: {ruta.resolve()}"
        )

    df = _leer_csv(ruta)

    # Verificar que estén todas las variables predictoras.
    faltantes = [
        col for col in FEATURES
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Al lote de inferencia le faltan variables: {faltantes}"
        )

    # Validar target si el archivo lo contiene.
    _validar_target(df)

    return df.copy()


def main():

    df = cargar_datos()

    print("Dataset Default of Credit Card Clients")
    print("-" * 50)
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Nulos: {int(df.isna().sum().sum()):,}")
    print(f"Clientes únicos: {df['ID'].nunique():,}")
    print(f"Target: {TARGET}")
    print(f"Features: {len(FEATURES)}")
    print(f"Columnas: {list(df.columns)}")

    print("\nDistribución del target:")
    print(df[TARGET].value_counts())

    print("\nPrimeras observaciones:")
    print(df.head())


if __name__ == "__main__":
    main()
