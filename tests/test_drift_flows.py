
import pandas as pd

from domain.modelo import VARIABLES
from infraestructure.flows import drift_features_flow as features_flow
from infraestructure.flows import drift_evento_flow as evento_flow


# --------------------------------------------------
# FEATURE DRIFT
# --------------------------------------------------


def test_feature_drift_guarda_reporte_html(monkeypatch, tmp_path):
    """Verifica que se genere el reporte HTML de Feature Drift."""

    class ResultadoFalso:
        def save_html(self, ruta):
            with open(ruta, "w", encoding="utf-8") as archivo:
                archivo.write("<html>Reporte de prueba</html>")

        def dict(self):
            return {
                "metrics": [
                    {
                        "metric_name": "DriftedColumnsCount",
                        "value": {"share": 0.5},
                    }
                ]
            }

    class ReporteFalso:
        def __init__(self, presets):
            pass

        def run(self, reference_data, current_data):
            # Verificar que ambos conjuntos tengan las mismas columnas.
            assert list(reference_data.columns) == list(
                current_data.columns
            )

            return ResultadoFalso()

    # Simular el comportamiento de Evidently.
    monkeypatch.setattr(
        features_flow,
        "Report",
        ReporteFalso,
    )

    # Crear lote de prueba con las 23 variables predictoras.
    datos = pd.DataFrame({
        columna: [100, 200]
        for columna in VARIABLES
    })

    ruta_lote = tmp_path / "lote.csv"
    datos.to_csv(ruta_lote, index=False)

    # Simular la carga del lote de inferencia.
    monkeypatch.setattr(
        features_flow,
        "cargar_lote",
        lambda ruta: datos.copy(),
    )

    # Simular la extracción de la fracción de variables con drift.
    monkeypatch.setattr(
        features_flow,
        "fraccion_desde_evidently",
        lambda resultado: resultado["metrics"][0]["value"]["share"],
    )

    cfg = {
        "monitoreo": {
            "carpeta_reportes": str(tmp_path / "reports"),
        }
    }

    # Ejecutar la tarea de cálculo de drift.
    fraccion = features_flow.calcular_drift.fn(
        datos,
        str(ruta_lote),
        cfg,
    )

    # Validar la fracción calculada.
    assert fraccion == 0.5

    # Validar que se haya generado el reporte HTML.
    reporte = (
        tmp_path
        / "reports"
        / "drift_features_lote.html"
    )

    assert reporte.exists()


# --------------------------------------------------
# TARGET DRIFT
# --------------------------------------------------


def test_target_drift_evalua_lote(monkeypatch):
    """Verifica que se calcule el drift de la variable objetivo."""

    lote = pd.DataFrame({
        "default_payment_next_month": [0, 0, 1, 0, 1, 0],
    })

    # Simular la carga del lote de inferencia.
    monkeypatch.setattr(
        evento_flow,
        "cargar_lote",
        lambda ruta: lote,
    )

    referencia = pd.Series(
        [0, 0, 0, 0, 1, 0],
        name="default_payment_next_month",
    )

    cfg = {
        "monitoreo": {
            "alfa_evento": 0.05,
        }
    }

    # Ejecutar la evaluación del drift de la variable objetivo.
    resultado = evento_flow.evaluar_evento.fn(
        referencia,
        "lote_prueba.csv",
        cfg,
    )

    # Validar que se retornen los indicadores esperados.
    assert "tasa_referencia" in resultado
    assert "tasa_lote" in resultado
    assert "p_valor" in resultado
    assert "drift" in resultado


def test_target_drift_genera_alerta():
    """Verifica la alerta cuando se detecta drift."""

    resultado = {
        "drift": True,
        "p_valor": 0.01,
        "tasa_referencia": 0.20,
        "tasa_lote": 0.60,
    }

    alerta = evento_flow.alerta.fn(
        resultado,
        alfa=0.05,
    )

    assert alerta is True


def test_target_drift_no_genera_alerta():
    """Verifica que no se genere alerta sin drift."""

    resultado = {
        "drift": False,
        "p_valor": 0.45,
        "tasa_referencia": 0.22,
        "tasa_lote": 0.24,
    }

    alerta = evento_flow.alerta.fn(
        resultado,
        alfa=0.05,
    )

    assert alerta is False
