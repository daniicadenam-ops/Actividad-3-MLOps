
"""Lógica de negocio del modelo Credit Default: variables,
pipeline de preprocesamiento, codificación y métricas.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Variables del modelo.
# ID es identificador y no se utiliza como predictor.

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

# Variables categóricas: se codifican con One-Hot Encoder.
CATEGORICAS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
]

VARIABLES = NUMERICAS + CATEGORICAS

OBJETIVO = "default_payment_next_month"


def resolver_features(columnas):
    """
    Convierte `columnas` de config.yaml ('ALL' o lista)
    en (numericas, categoricas).
    """

    if isinstance(columnas, str) and columnas.upper() == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [
            v for v in columnas
            if v not in VARIABLES
        ]

        if desconocidas:
            raise ValueError(
                f"variables desconocidas en 'columnas': {desconocidas}"
            )

        elegidas = columnas

    if not elegidas:
        raise ValueError(
            "la selección de features no tiene ninguna variable"
        )

    numericas = [
        v for v in NUMERICAS
        if v in elegidas
    ]

    categoricas = [
        v for v in CATEGORICAS
        if v in elegidas
    ]

    return numericas, categoricas


def construir_modelo(
    tipo,
    parametros,
    numericas,
    categoricas
):
    """
    Construye un pipeline compuesto por:

    1. Estandarización de variables numéricas.
    2. Codificación One-Hot de variables categóricas.
    3. Entrenamiento del estimador seleccionado.

    El preprocesamiento se ajusta únicamente con los datos
    de entrenamiento cuando se ejecuta fit().
    """

    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)

    elif tipo == "gradient_boosting":
        estimador = GradientBoostingClassifier(**parametros)

    elif tipo == "random_forest":
        estimador = RandomForestClassifier(**parametros)

    else:
        raise ValueError(
            f"modelo '{tipo}' no soportado"
        )

    transformadores = []

    # Preprocesamiento de variables numéricas.
    if numericas:
        transformadores.append(
            ("num", StandardScaler(), numericas)
        )

    # Codificación de variables categóricas.
    if categoricas:
        transformadores.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categoricas,
            )
        )

    preprocesado = ColumnTransformer(
        transformers=transformadores,
        remainder="drop",
    )

    return Pipeline([
        ("preprocesado", preprocesado),
        ("modelo", estimador),
    ])


def calcular_metricas(y_true, proba, umbral):
    """
    Métricas de clasificación binaria a partir
    de la probabilidad de incumplimiento.
    """

    pred = (proba >= umbral).astype(int)

    return {
        "auc": float(
            roc_auc_score(y_true, proba)
        ),
        "accuracy": float(
            accuracy_score(y_true, pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0
            )
        ),
    }
