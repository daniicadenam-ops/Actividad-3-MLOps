
"""
Actividad 3: Monitoreo y simulación de Drift.

Proyecto: Credit Default

Incluye:
- División de datos de referencia.
- Simulación de drift en variables de entrada.
- Simulación de drift en la variable objetivo.
- Validación de lotes.
- Detección de Target Drift mediante chi-cuadrado.
- Extracción de resultados de Data Drift desde Evidently.
"""

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURACIÓN DE VARIABLES
# ============================================================

# Variable objetivo
OBJETIVO = "default_payment_next_month"

# Variables numéricas utilizadas para simular drift
NUMERICAS = [
    "LIMIT_BAL",
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
]

# Todas las variables predictoras del modelo
# Se excluyen ID y la variable objetivo
VARIABLES = [
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
]

# Subida máxima de la tasa de incumplimiento.
# 0.3 representa un incremento de 30 puntos porcentuales.
SUBIDA_TASA_MAXIMA = 0.3


# ============================================================
# 2. DIVISIÓN DE DATOS
# ============================================================

def dividir(df, division):
    """
    Divide los datos igual que el entrenamiento.

    Devuelve:
        entrenamiento, prueba
    """

    return train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )


# ============================================================
# 3. SIMULACIÓN DE DRIFT
# ============================================================

def aplicar_drift(df, nivel, semilla=42):
    """
    Simula drift en un lote de Credit Default.

    Aplica:
    1. Drift en variables numéricas.
    2. Drift en la variable objetivo.

    nivel:
        Valor entre 0 y 1 que representa la intensidad
        de la simulación.
    """

    if not 0 <= nivel <= 1:
        raise ValueError(
            f"El nivel de drift debe estar entre 0 y 1, no {nivel}"
        )

    # Verificar que existan las variables necesarias
    columnas_faltantes = [
        columna
        for columna in NUMERICAS
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan variables numéricas para simular drift: "
            f"{columnas_faltantes}"
        )

    out = df.copy()
    rng = np.random.default_rng(semilla)

    # --------------------------------------------------------
    # 3.1 Drift en variables numéricas
    # --------------------------------------------------------

    n = round(nivel * len(NUMERICAS))

    if n > 0:

        columnas_afectadas = rng.choice(
            NUMERICAS,
            size=n,
            replace=False,
        )

        for columna in columnas_afectadas:

            std_val = out[columna].std()

            if pd.notna(std_val) and std_val > 0:

                # Desplazamiento de dos desviaciones estándar
                out[columna] = (
                    out[columna] + 2 * std_val
                )

    # --------------------------------------------------------
    # 3.2 Drift en la variable objetivo
    # --------------------------------------------------------

    if OBJETIVO in out.columns:

        out[OBJETIVO] = _subir_tasa(
            out[OBJETIVO],
            nivel * SUBIDA_TASA_MAXIMA,
            rng,
        )

    return out


def _subir_tasa(objetivo, subida, rng):
    """
    Convierte registros negativos (0) en positivos (1)
    para aumentar artificialmente la tasa de incumplimiento.
    """

    objetivo_int = objetivo.astype(int)

    if not objetivo_int.isin([0, 1]).all():
        raise ValueError(
            f"La variable {OBJETIVO} debe contener únicamente 0 y 1"
        )

    tasa_actual = objetivo_int.mean()

    nueva_tasa = min(
        tasa_actual + subida,
        1.0,
    )

    pos_objetivo = round(
        nueva_tasa * len(objetivo_int)
    )

    faltan = (
        pos_objetivo - int(objetivo_int.sum())
    )

    if faltan <= 0:
        return objetivo

    subidos = objetivo.copy()

    indices_ceros = objetivo_int.index[
        objetivo_int == 0
    ]

    faltan = min(
        faltan,
        len(indices_ceros),
    )

    if faltan > 0:

        a_cambiar = rng.choice(
            indices_ceros.to_numpy(),
            size=faltan,
            replace=False,
        )

        subidos.loc[a_cambiar] = 1

    return subidos


# ============================================================
# 4. RESULTADOS DE EVIDENTLY
# ============================================================

def fraccion_desde_evidently(resultado):
    """
    Obtiene la fracción de variables con drift
    desde el resultado de Evidently.
    """

    for metrica in resultado["metrics"]:

        if metrica["metric_name"].startswith(
            "DriftedColumnsCount"
        ):

            return float(
                metrica["value"]["share"]
            )

    raise ValueError(
        "El resultado de Evidently no incluye DriftedColumnsCount"
    )


# ============================================================
# 5. VALIDACIÓN DEL LOTE
# ============================================================

def validar_lote(lote):
    """
    Valida que el lote tenga la variable objetivo,
    sin valores vacíos y con valores binarios.
    """

    if OBJETIVO not in lote.columns:

        raise ValueError(
            f"El lote no tiene la columna '{OBJETIVO}': "
            "no se puede calcular el drift de evento"
        )

    if lote[OBJETIVO].isna().any():

        raise ValueError(
            f"El lote tiene '{OBJETIVO}' sin valor: "
            "no se puede calcular el drift de evento"
        )

    if not lote[OBJETIVO].isin([0, 1]).all():

        raise ValueError(
            f"La variable '{OBJETIVO}' debe contener únicamente 0 y 1"
        )


# ============================================================
# 6. DETECCIÓN DE TARGET DRIFT
# ============================================================

def drift_de_target(referencia, lote, alfa):
    """
    Compara la proporción de incumplimiento entre
    el conjunto de referencia y el lote actual.

    Utiliza la prueba chi-cuadrado.

    Devuelve:
        tasa_referencia
        tasa_lote
        p_valor
        drift
    """

    ref_int = referencia.astype(int)
    lote_int = lote.astype(int)

    if len(ref_int) == 0 or len(lote_int) == 0:
        raise ValueError(
            "Los conjuntos de referencia y lote no pueden estar vacíos"
        )

    if not ref_int.isin([0, 1]).all():
        raise ValueError(
            "La referencia debe contener únicamente 0 y 1"
        )

    if not lote_int.isin([0, 1]).all():
        raise ValueError(
            "El lote debe contener únicamente 0 y 1"
        )

    # Conteos de incumplimiento y no incumplimiento
    pos_ref = int(ref_int.sum())
    neg_ref = len(ref_int) - pos_ref

    pos_lote = int(lote_int.sum())
    neg_lote = len(lote_int) - pos_lote

    tabla = [
        [pos_ref, neg_ref],
        [pos_lote, neg_lote],
    ]

    # Prueba estadística chi-cuadrado
    _, p_valor, _, _ = chi2_contingency(tabla)

    return {
        "tasa_referencia": pos_ref / len(ref_int),
        "tasa_lote": pos_lote / len(lote_int),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }
