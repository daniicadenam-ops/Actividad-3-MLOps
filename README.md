# Actividad 3 -- MLOps: Monitoreo y emulación de Drift (Credit Default)

## 1. Descripción del proyecto

Esta actividad amplía la solución MLOps de Credit Default de la
Actividad 2. Incorpora emulación controlada de Drift en la inferencia,
monitoreo de las variables predictoras (Feature Drift), monitoreo de la
variable objetivo (Target Drift) y generación de alertas mediante
umbrales configurables.

Se utiliza el dataset **Default of Credit Card Clients** y la variable
objetivo `default_payment_next_month`: - `0`: no presenta incumplimiento
el siguiente mes. - `1`: presenta incumplimiento el siguiente mes.

La emulación es un mecanismo experimental para probar el monitoreo; no
representa necesariamente un cambio real observado en producción.

## 2. Dataset y variables

Archivo esperado:

``` text
data/default of credit card clients.csv
```

Se excluye `ID` de las variables predictoras. El monitoreo de Features
evalúa las 23 variables utilizadas por el modelo: - Información del
cliente: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`. - Historial
de pagos: `PAY_0`, `PAY_2` a `PAY_6`. - Saldos facturados: `BILL_AMT1` a
`BILL_AMT6`. - Pagos realizados: `PAY_AMT1` a `PAY_AMT6`.

El monitoreo de Target utiliza `default_payment_next_month` cuando las
etiquetas reales están disponibles.

## 3. Tecnologías

-   **Python:** lenguaje principal.
-   **Pandas / NumPy:** preparación, manipulación y emulación de datos.
-   **Scikit-learn:** división estratificada de los datos.
-   **Evidently:** detección de Feature Drift y reportes HTML.
-   **SciPy:** prueba Chi-cuadrado para Target Drift.
-   **Prefect:** orquestación de los flujos.
-   **MLflow:** seguimiento de ejecuciones, modelo y artefactos.
-   **Pytest:** pruebas automatizadas.
-   **Poetry:** gestión de dependencias y entorno.

## 4. Estructura relevante del repositorio

``` text
Actividad-3-MLOps/
├── configs/
│   └── config.yaml
├── data/
│   ├── default of credit card clients.csv
│   └── predicciones/
├── domain/
│   ├── modelo.py
│   └── monitoreo.py
├── infraestructure/
│   ├── api/
│   │   └── main.py
│   └── flows/
│       ├── entrenamiento_flow.py
│       ├── inferencia_flow.py
│       ├── drift_features_flow.py
│       ├── drift_evento_flow.py
│       └── desplegar.py
├── reports/
│   └── drift/
├── scripts/
│   └── load_data.py
├── tests/
│       ├── test_api.py
│       ├── test_datos.py
│       ├── test_drift_flows.py
│       ├── test_inferencia.py
│       ├── test_modelo.py
│       └── test_monitoreo.py
├── services/
│   └── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md
```

La estructura es orientativa y refleja los componentes principales de la
actividad; se deben conservar los demás archivos de la Actividad 2
presentes en el repositorio.

## 5. Requisitos previos

-   Python compatible con `pyproject.toml`.
-   Poetry instalado.
-   Dataset disponible en la ruta configurada.
-   Docker Desktop, si se utilizan los servicios de infraestructura.
-   Prefect y MLflow disponibles para los flujos que los utilizan.

Instalar dependencias:

``` powershell
poetry install
```

Si se utiliza Docker Compose, desde `services/`:

``` powershell
docker compose up -d
docker compose ps
```

Detener servicios:

``` powershell
docker compose down
```

## 6. Configuración

La configuración central se encuentra en `configs/config.yaml`.

### 6.1 Emulación de Drift

``` yaml
inferencia:
  tamano_lote: 100
  carpeta_salida: "data/predicciones"
  drift:
    nivel: 0.0
    semilla: 42
```

Niveles permitidos:

  Nivel   Interpretación
  ------- ----------------------------------------------
  `0.0`   Escenario base, sin perturbación artificial.
  `0.3`   Perturbación moderada.
  `0.6`   Perturbación alta.

En la implementación de `domain/monitoreo.py`, `emular_drift()`
multiplica por `1 + nivel` las columnas financieras definidas en
`COLUMNAS_DRIFT`. Por tanto, 0.3 aplica un factor de 1.3 y 0.6 un factor
de 1.6. La función actual no utiliza el parámetro `semilla`.

El nivel de emulación no es el porcentaje de columnas detectadas con
Drift. Ese porcentaje es un resultado calculado por Evidently.

### 6.2 Parámetros de monitoreo

``` yaml
monitoreo:
  alfa_evento: 0.05
  umbral_drift: 0.30
  carpeta_reportes: "reports/drift"
```

-   `alfa_evento`: nivel de significancia para Target Drift.
-   `umbral_drift`: proporción máxima permitida de columnas con Feature
    Drift.
-   `carpeta_reportes`: carpeta donde se guardan los reportes HTML.

El flujo de Features construye automáticamente la ruta del CSV según
`inferencia.drift.nivel` y la fecha actual. Se debe ejecutar primero la
inferencia del escenario y después el flujo de Features.

**Compatibilidad del flujo Target:** revisar `drift_evento_flow.py` para
confirmar cómo obtiene la ruta del lote. Si el código consulta
`cfg["monitoreo"]["lote_actual"]`, se debe conservar esa clave en el
YAML y actualizarla a la ruta del CSV que se desea evaluar. La selección
automática descrita arriba corresponde al flujo de Features.

## 7. Emulación de Drift en inferencia

El flujo `infraestructure/flows/inferencia_flow.py` carga el modelo con
alias `champion` desde MLflow, ingesta un lote, aplica la emulación
configurada, genera predicciones y guarda el CSV. También registra la
ejecución y el nivel de Drift en MLflow.

Ejecución desde la raíz:

``` powershell
poetry run python infraestructure/flows/inferencia_flow.py
```

Los archivos se guardan en `data/predicciones/` con el nivel y la fecha
en el nombre:

``` text
predicciones_credit_default_drift_0_0_FECHA.csv
predicciones_credit_default_drift_0_3_FECHA.csv
predicciones_credit_default_drift_0_6_FECHA.csv
```

`FECHA` representa la fecha local de ejecución.

## 8. Monitoreo de Feature Drift

Archivo:

``` text
infraestructure/flows/drift_features_flow.py
```

El flujo: 1. Lee la configuración. 2. Identifica el nivel de Drift. 3.
Selecciona el CSV de inferencia correspondiente al nivel y fecha actual.
4. Carga la referencia, obtenida de la partición de entrenamiento. 5.
Selecciona las 23 variables predictoras en referencia y lote actual. 6.
Ejecuta `Report([DataDriftPreset()])` de Evidently. 7. Guarda el reporte
HTML. 8. Extrae la proporción de columnas con Drift y la compara con el
umbral. 9. Genera una alerta cuando la proporción supera el umbral.

Ejecución:

``` powershell
poetry run python infraestructure/flows/drift_features_flow.py
```

Los reportes se guardan en `reports/drift/`, con nombres diferenciados
por lote, por ejemplo:

``` text
drift_features_predicciones_credit_default_drift_0_0_FECHA.html
drift_features_predicciones_credit_default_drift_0_3_FECHA.html
drift_features_predicciones_credit_default_drift_0_6_FECHA.html
```

### Interpretación

La proporción de columnas con Drift se calcula sobre las variables
predictoras evaluadas. Por ejemplo, `0.43478` equivale aproximadamente
al 43.478 % de las columnas. No significa que ese porcentaje de clientes
haya cambiado ni que el desempeño del modelo haya disminuido en esa
proporción.

La condición implementada para la alerta es: - Si la proporción es mayor
que `umbral_drift`, se genera alerta. - Si es menor o igual, no se
genera alerta.

## 9. Monitoreo de Target Drift

Archivo:

``` text
infraestructure/flows/drift_evento_flow.py
```

El flujo compara la proporción de incumplimiento de la referencia con la
del lote actual usando `default_payment_next_month` y una prueba
Chi-cuadrado.

Ejecución:

``` powershell
poetry run python infraestructure/flows/drift_evento_flow.py
```

El proceso valida la variable objetivo, calcula las tasas de
incumplimiento, construye una tabla de contingencia, ejecuta
Chi-cuadrado y compara el valor p con `monitoreo.alfa_evento`.

La regla de decisión es: - Si `p-valor < alfa_evento`, se considera
evidencia estadística de cambio y se genera alerta. - En caso contrario,
no se genera alerta.

El monitoreo del Target requiere etiquetas reales. En producción, estas
etiquetas estarán disponibles cuando se conozca el resultado efectivo de
pago. Las predicciones del modelo no deben sustituir al target real.

## 10. Orden recomendado de ejecución

Para cada escenario, cambiar `inferencia.drift.nivel` en `config.yaml` y
ejecutar primero inferencia y luego Feature Drift.

### Escenario 0.0

``` yaml
drift:
  nivel: 0.0
```

``` powershell
poetry run python infraestructure/flows/inferencia_flow.py
poetry run python infraestructure/flows/drift_features_flow.py
```

### Escenario 0.3

``` yaml
drift:
  nivel: 0.3
```

``` powershell
poetry run python infraestructure/flows/inferencia_flow.py
poetry run python infraestructure/flows/drift_features_flow.py
```

### Escenario 0.6

``` yaml
drift:
  nivel: 0.6
```

``` powershell
poetry run python infraestructure/flows/inferencia_flow.py
poetry run python infraestructure/flows/drift_features_flow.py
```

Para monitorear Target, ejecutar además `drift_evento_flow.py` con la
ruta del lote configurada de acuerdo con la implementación de ese flujo.

**Criterio de comparación:** los reportes de 0.0 y 0.6 deben presentar
resultados distintos atribuibles a la perturbación. Mantener la misma
referencia, tamaño de lote y umbral para que la comparación sea válida.
No cambiar el umbral solo para forzar una alerta.

## 11. Pruebas automatizadas

Ejecutar desde la raíz:

``` powershell
poetry run pytest -v
```

Pruebas específicas de monitoreo:

``` powershell
poetry run pytest tests/test_monitoreo.py tests/test_drift_flows.py -v
```

Pruebas de inferencia:

``` powershell
poetry run pytest tests/test_inferencia_flow.py -v
```

Las pruebas cubren los componentes que estén presentes en `tests/`:
datos, modelo, inferencia, API y monitoreo de Drift. Registrar como
evidencia la salida real de pytest; no asumir que todas las pruebas
pasan sin ejecutar el comando.

## 12. Resultados documentados

En una ejecución de Feature Drift se obtuvo:

  Indicador                            Resultado
  ---------------------------------- -----------
  Variables evaluadas                         23
  Variables con Drift                         10
  Variables sin Drift                         13
  Proporción de columnas con Drift      43.478 %

Este resultado corresponde al lote y referencia de esa ejecución; no
debe asumirse que será igual para todos los niveles de emulación.

En una ejecución de Target Drift se obtuvo:

  Indicador                                                                            Resultado
  -------------------------------------- -------------------------------------------------------
  Tasa de incumplimiento de referencia                                                   22.12 %
  Tasa de incumplimiento del lote                                                        26.00 %
  Valor p                                                                               0.416944
  Nivel de significancia                                                                    0.05
  Decisión                                 No se identificó Drift estadísticamente significativo
  Alerta                                                                     No se generó alerta

Estos valores son resultados observados en ejecuciones concretas, no
resultados garantizados para cualquier lote.


