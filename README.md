# Sleep Prediction

Proyecto de análisis y modelado para **predecir Sleep Duration** (horas de sueño) a partir de variables de estilo de vida usando el dataset *Sleep Health and Lifestyle*.

## Dataset

Los datos se obtienen de Kaggle:

**URL del dataset:**  
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

Descarga el CSV y colócalo en la raíz del proyecto como `sleep_health_and_lifestyle.csv` o `Sleep_health_and_lifestyle_dataset.csv` (el notebook intenta cargar ambos nombres).

## Contenido

- **`sleep_prediction_notebook.ipynb`**: notebook completo con carga de datos, limpieza, EDA, pipeline de modelado (Ridge, RandomForest, HistGradientBoosting/XGBoost), interpretación y demo con perfiles ficticios.

## Requisitos

- Python 3
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- Opcional: `xgboost`

## Uso

1. Clona el repositorio y descarga el dataset desde la URL anterior.
2. Coloca el CSV en la carpeta del proyecto.
3. Abre y ejecuta `sleep_prediction_notebook.ipynb`.
