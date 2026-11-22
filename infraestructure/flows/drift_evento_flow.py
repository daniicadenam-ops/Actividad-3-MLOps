
"""
Actividad 3: monitoreo de drift de evento (Target Drift).

Compara la proporción de incumplimiento de pago del lote
con la proporción del conjunto de entrenamiento.

Proyecto: Credit Default
Variable objetivo: default_payment_next_month
"""

import yaml

from prefect import flow, task

from domain.monitoreo import (
    OBJETIVO,
    dividir,
    drift_de_target,
    validar_lote,
)

from scripts.load_data import cargar_datos, cargar_lote


CONFIG = "configs/config.yaml"


@task
def referencia(cfg):
    """
    Obtiene la variable objetivo del conjunto de entrenamiento.
    """

    df = cargar_datos(cfg["datos"]["ruta"])

    entrenamiento, _ = dividir(
        df,
        cfg["division"],
    )

    print(
        f"Referencia cargada: {len(entrenamiento)} registros"
    )

    print(
        f"Tasa de incumplimiento en entrenamiento: "
        f"{entrenamiento[OBJETIVO].mean():.2%}"
    )

    return entrenamiento[OBJETIVO]


@task
def evaluar_evento(referencia_objetivo, ruta_lote, cfg):
    """
    Compara la tasa de incumplimiento del lote
    con la referencia del entrenamiento.
    """

    lote = cargar_lote(ruta_lote)

    # Validar que el lote tenga etiquetas reales.
    validar_lote(lote)

    resultado = drift_de_target(
        referencia_objetivo,
        lote[OBJETIVO],
        cfg["monitoreo"]["alfa_evento"],
    )

    print("\n--- Resultado del monitoreo de Target Drift ---")

    print(
        f"Tasa de incumplimiento en entrenamiento: "
        f"{resultado['tasa_referencia']:.2%}"
    )

    print(
        f"Tasa de incumplimiento en el lote: "
        f"{resultado['tasa_lote']:.2%}"
    )

    print(
        f"P-valor: {resultado['p_valor']:.6f}"
    )

    return resultado


@task
def alerta(resultado, alfa):
    """
    Genera una alerta cuando existe un cambio
    estadísticamente significativo.
    """

    hay_alerta = resultado["drift"]

    if hay_alerta:
        print(
            f"ALERTA: se detectó drift en {OBJETIVO}. "
            f"P-valor {resultado['p_valor']:.6f} < {alfa}"
        )
    else:
        print(
            f"Sin alerta: no se detectó drift significativo "
            f"en {OBJETIVO}. "
            f"P-valor {resultado['p_valor']:.6f} >= {alfa}"
        )

    return hay_alerta


@flow(name="drift-evento-credit-default", log_prints=True)
def drift_evento_flow():

    with open(CONFIG, encoding="utf-8") as archivo:
        cfg = yaml.safe_load(archivo)

    referencia_objetivo = referencia(cfg)

    resultado = evaluar_evento(
        referencia_objetivo,
        cfg["monitoreo"]["lote_actual"],
        cfg,
    )

    return alerta(
        resultado,
        cfg["monitoreo"]["alfa_evento"],
    )


if __name__ == "__main__":
    drift_evento_flow()
