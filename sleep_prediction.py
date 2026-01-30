"""
Sleep Prediction - Script ejecutable
Convertido desde sleep_prediction_notebook.ipynb
Predice Sleep Duration a partir de hábitos (dataset Sleep Health and Lifestyle).
"""

# ¿Se pueden predecir tus horas de sueño con hábitos?

## Objetivo del vídeo

# En este notebook vamos a **analizar y modelar** el dataset "Sleep Health and Lifestyle" (Kaggle) para **predecir Sleep Duration** (horas de sueño) a partir de variables de estilo de vida.

# **Qué vamos a ver:**
# - Variables típicas: estrés, actividad física, edad, BMI, ocupación, género (no prometemos magia, solo patrones).
# - **Target:** Sleep Duration → problema de **regresión** (predecir un número).
# - Flujo: datos → limpieza → EDA → pipeline → modelos → interpretación → demo con perfiles ficticios.

# ---
# *Nota para el vídeo:* "¿Sabías que tus hábitos pueden estar relacionados con cuánto duermes? Hoy vamos a usar datos reales para intentar predecir horas de sueño. Sin consejos médicos: solo ciencia de datos."

## 1. Carga y vistazo rápido

# **Qué vamos a ver:**
# - Leer el CSV y comprobar dimensiones.
# - Ver columnas, primeras filas y % de nulos.
# - Identificar el target (Sleep Duration) y columnas disponibles para features.

# Antes de modelar, hay que **entender qué datos tenemos**.

# ---
# *Nota para el vídeo:* "Antes de tocar ningún modelo, miramos los datos: cuántas filas, qué columnas y si hay huecos. Es el paso que muchos se saltan y luego se quejan de que el modelo no entiende nada."

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar CSV (prueba dos nombres típicos del dataset)
csv_names = ["sleep_health_and_lifestyle.csv", "Sleep_health_and_lifestyle_dataset.csv"]
df = None
for name in csv_names:
    path = Path(name)
    if path.exists():
        df = pd.read_csv(path)
        print(f"Archivo cargado: {name}")
        break
if df is None:
    raise FileNotFoundError("No se encontró el CSV. Coloca sleep_health_and_lifestyle.csv (o el .csv del dataset) en la carpeta del notebook.")

print("Shape:", df.shape)
print("\nColumnas:", list(df.columns))
print(df.head())

# Porcentaje de nulos por columna
nulos = (df.isnull().sum() / len(df) * 100).round(2)
print("Nulos por columna (%):")
print(nulos[nulos > 0] if nulos.sum() > 0 else "No hay nulos.")
print(nulos)

# Identificar target: Sleep Duration (o nombre parecido)
target_candidates = [c for c in df.columns if "sleep" in c.lower() and "duration" in c.lower()]
if not target_candidates:
    target_candidates = [c for c in df.columns if "duration" in c.lower()]
TARGET = target_candidates[0] if target_candidates else "Sleep Duration"
print("Target elegido:", TARGET)
print(df[TARGET].describe())

## 2. Limpieza mínima y preparación

# **Qué vamos a ver:**
# - Quitar duplicados.
# - Nulos: numéricas → mediana; categóricas → "Unknown".
# - Separar numéricas vs categóricas y elegir features (edad, actividad, estrés, BMI, género, ocupación, etc.).
# - Descartar columnas que no aportan (ID, nombre).

# Menos es más: una limpieza **simple y reproducible**.

# ---
# *Nota para el vídeo:* "Menos es más: eliminamos duplicados, rellenamos huecos con la mediana o 'Unknown', y nos quedamos solo con columnas que tienen sentido para predecir sueño."

# Duplicados
before = len(df)
df = df.drop_duplicates()
print(f"Duplicados eliminados: {before - len(df)}. Filas restantes: {len(df)}")

# Columnas a descartar (IDs, nombres, etc.)
drop_patterns = ["person id", "id", "name"]
to_drop = [c for c in df.columns if any(p in c.lower() for p in drop_patterns)]
df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")
print("Columnas descartadas:", to_drop if to_drop else "ninguna por patrón ID/name")

# Tipos: numéricas vs categóricas (excluyendo target)
def get_feature_columns(df, target):
    exclude = {target, "Person ID", "Person id", "person id"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    # Excluir Quality of Sleep para evitar fugas (muy ligada al target)
    if "Quality of Sleep" in numeric_cols:
        numeric_cols.remove("Quality of Sleep")
    cat_cols = [c for c in df.columns if c not in exclude and c not in numeric_cols and df[c].dtype == object]
    # Blood Pressure tipo "126/83" no la usamos como numérica aquí
    if "Blood Pressure" in numeric_cols:
        numeric_cols.remove("Blood Pressure")
    return numeric_cols, cat_cols

numeric_cols, cat_cols = get_feature_columns(df, TARGET)
print("Numéricas (features):", numeric_cols)
print("Categóricas:", cat_cols)

# Imputar nulos: numéricas → mediana, categóricas → "Unknown"
for c in numeric_cols:
    if c in df.columns and df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if c in df.columns and df[c].isnull().any():
        df[c] = df[c].fillna("Unknown")
print("Limpieza aplicada. Nulos restantes:", df[numeric_cols + cat_cols + [TARGET]].isnull().sum().sum())

FEATURE_COLS = [c for c in (numeric_cols + cat_cols) if c in df.columns]
print("Features finales:", FEATURE_COLS)

## 3. EDA express (insights en minutos)

# **Qué vamos a ver:**
# - Histograma del target (Sleep Duration).
# - Boxplot Sleep Duration por Stress Level.
# - Scatter: Physical Activity vs Sleep Duration.
# - Comparación por Gender o BMI Category.
# - Heatmap de correlación (numéricas).

# Patrones visibles > 20 gráficas.

# ---
# *Nota para el vídeo:* "Solo unas pocas gráficas que cuenten historia: distribución del sueño, cómo se relaciona con estrés y actividad, y correlaciones. Lo que vemos aquí nos guiará la interpretación del modelo."

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(df[TARGET], bins=20, color="steelblue", edgecolor="white")
ax.set_xlabel("Sleep Duration (horas)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución del target: Sleep Duration")
plt.tight_layout()
plt.show()

# *Conclusión:* La distribución es aproximadamente unimodal; la mayoría duerme entre 6 y 8 horas. Nos sirve para ver que no hay valores extremos raros.

# Boxplot: Sleep Duration por Stress Level (o columna similar)
stress_col = "Stress Level" if "Stress Level" in df.columns else next((c for c in df.columns if "stress" in c.lower()), None)
if stress_col is not None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    levels = sorted(df[stress_col].dropna().unique())
    data_by_level = [df[df[stress_col] == l][TARGET].dropna() for l in levels]
    ax.boxplot(data_by_level, labels=[str(l) for l in levels])
    ax.set_xlabel(stress_col)
    ax.set_ylabel("Sleep Duration (horas)")
    ax.set_title(f"Sleep Duration por {stress_col}")
    plt.tight_layout()
    plt.show()

# *Conclusión:* A mayor nivel de estrés suele bajar la duración del sueño; es un candidato fuerte como predictor.

# Scatter: Physical Activity vs Sleep Duration
act_col = next((c for c in df.columns if "physical" in c.lower() or "activity" in c.lower()), None)
if act_col and act_col in df.columns:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df[act_col], df[TARGET], alpha=0.5, s=20)
    ax.set_xlabel(act_col)
    ax.set_ylabel("Sleep Duration (horas)")
    ax.set_title("Actividad física vs Sleep Duration")
    plt.tight_layout()
    plt.show()

# *Conclusión:* Más actividad física suele asociarse con más horas de sueño en este dataset; refuerza la idea de usarla como feature.

# Comparación por Gender o BMI Category
cat_plot = "Gender" if "Gender" in df.columns else ("BMI Category" if "BMI Category" in df.columns else None)
if cat_plot:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    df.boxplot(column=TARGET, by=cat_plot, ax=ax)
    ax.set_xlabel(cat_plot)
    ax.set_ylabel("Sleep Duration (horas)")
    ax.set_title(f"Sleep Duration por {cat_plot}")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()

# Heatmap de correlación (solo numéricas)
num_for_corr = [c for c in numeric_cols + [TARGET] if c in df.columns]
corr = df[num_for_corr].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
plt.colorbar(im, ax=ax)
ax.set_title("Correlación (numéricas)")
plt.tight_layout()
plt.show()

# *Conclusión:* Las variables más correlacionadas con Sleep Duration son las que usaremos en el modelo; el heatmap ayuda a no duplicar información (multicolinealidad).

## 4. Pipeline de modelado

# **Qué vamos a ver:**
# - Separar X (features) e y (target).
# - Train/test split.
# - ColumnTransformer: numéricas → SimpleImputer + StandardScaler; categóricas → SimpleImputer + OneHotEncoder.
# - Métricas: MAE (error medio absoluto en horas) y RMSE (penaliza más los errores grandes).

# Esto evita fugas de información y hace el flujo **replicable**.

# ---
# *Nota para el vídeo:* "Usamos un pipeline con ColumnTransformer: imputamos y escalamos numéricas, codificamos categóricas. Así no hay fugas entre train y test y cualquiera puede reproducir el resultado."

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", len(X_train), "| Test size:", len(X_test))

num_cols = [c for c in FEATURE_COLS if c in numeric_cols]
cat_cols_used = [c for c in FEATURE_COLS if c in cat_cols]
transformers = [
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num_cols),
]
if cat_cols_used:
    transformers.append(("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols_used))
preprocessor = ColumnTransformer(transformers, remainder="drop")
print("Preprocesador listo. Numéricas:", num_cols, "| Categóricas:", cat_cols_used)

# **MAE:** error medio absoluto en horas (ej. MAE=0.3 → ~18 min de error medio).  
# **RMSE:** raíz del error cuadrático medio; penaliza más los errores grandes.

## 5. Comparativa rápida de modelos

# **Qué vamos a ver:**
# - Entrenar 3–4 modelos con el mismo pipeline: Ridge, RandomForest, HistGradientBoosting y (opcional) XGBoost.
# - CV 5-fold en train (MAE medio).
# - Entrenar en train y evaluar en test (MAE y RMSE).
# - Tabla ordenada por MAE; elegir el mejor.

# No siempre gana el modelo más complejo.

# ---
# *Nota para el vídeo:* "Probamos varios modelos con el mismo pipeline. Al final comparamos MAE en test; a veces un modelo simple se defiende muy bien."

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

def make_pipeline(estimator):
    return Pipeline([("prep", preprocessor), ("model", estimator)])

def evaluate(name, pipe, cv=5):
    cv_mae = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return {"model": name, "CV_MAE": round(cv_mae, 4), "Test_MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)}

models = [
    ("Ridge", Ridge(alpha=1.0, random_state=42)),
    ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=100, random_state=42)),
]
try:
    from xgboost import XGBRegressor
    models.append(("XGBoost", XGBRegressor(n_estimators=100, max_depth=5, random_state=42)))
except ImportError:
    pass

results = []
for name, est in models:
    pipe = make_pipeline(est)
    results.append(evaluate(name, pipe))
res_df = pd.DataFrame(results).sort_values("Test_MAE")
print(res_df)

best_name = res_df.iloc[0]["model"]
best_estimator = next(est for name, est in models if name == best_name)
best_pipeline = make_pipeline(best_estimator)
best_pipeline.fit(X_train, y_train)
print("Mejor modelo (por Test MAE):", best_name)

## 6. Interpretación simple del mejor modelo

# **Qué vamos a ver:**
# - Si es árboles/boosting: **feature importance** (nombres correctos tras OneHotEncoder).
# - Si es lineal: coeficientes principales.
# - 2–3 conclusiones accionables solo si el modelo/EDA las respaldan.

# Interpretar sin vender humo.

# ---
# *Nota para el vídeo:* "Vemos qué variables pesan más en el mejor modelo: estrés, actividad, edad, etc. Solo sacamos conclusiones que los datos respaldan."

# Obtener nombres de features tras el preprocesador
prep = best_pipeline.named_steps["prep"]
model = best_pipeline.named_steps["model"]
feature_names = list(num_cols)
if cat_cols_used and "cat" in prep.named_transformers_:
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat_cols_used))
    feature_names = feature_names + cat_names

# Feature importance (árboles/boosting) o coeficientes (lineal)
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
elif hasattr(model, "coef_"):
    imp = np.abs(model.coef_).ravel()
else:
    imp = None
if imp is not None and len(imp) == len(feature_names):
    idx = np.argsort(imp)[::-1][:10]
    top_names = [feature_names[i] for i in idx]
    top_imp = imp[idx]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(top_names)), top_imp, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (abs)" if hasattr(model, "coef_") else "Importancia")
    ax.set_title(f"Top 10 features — {best_name}")
    plt.tight_layout()
    plt.show()

# **Conclusiones accionables (si el modelo/EDA lo respaldan):**
# - **Estrés alto** suele asociarse a menos horas de sueño → gestionar estrés puede ayudar.
# - **Más actividad física** (y/o más pasos) suele asociarse a más sueño → movimiento como factor positivo.
# - **Edad y BMI** pueden tener efecto; el modelo nos dice el peso relativo en este dataset, no en la vida real.

## 7. Demo final (la parte viral)

# **Qué vamos a ver:**
# - Tres perfiles ficticios: persona estresada + poca actividad; persona activa + estrés bajo; intermedio.
# - Pasar cada perfil por el pipeline del mejor modelo y mostrar predicción de horas.
# - Tabla resumen y disclaimer: no es consejo médico; es un ejemplo de patrones en datos.

# Esto es lo que la gente comparte.

# ---
# *Nota para el vídeo:* "Vamos a simular tres perfiles: alguien muy estresado y sedentario, alguien activo y relajado, y uno intermedio. El modelo predice cuántas horas dormirían. Esto es lo que la gente comparte en redes."

# Perfiles ficticios (columnas deben coincidir con FEATURE_COLS)
def build_profile(gender, age, occupation, physical_activity, stress_level, bmi_category, heart_rate, daily_steps, sleep_disorder="None", blood_pressure="120/80"):
    d = {}
    if "Gender" in FEATURE_COLS:
        d["Gender"] = gender
    if "Age" in FEATURE_COLS:
        d["Age"] = age
    if "Occupation" in FEATURE_COLS:
        d["Occupation"] = occupation
    if "Physical Activity Level" in FEATURE_COLS:
        d["Physical Activity Level"] = physical_activity
    if "Stress Level" in FEATURE_COLS:
        d["Stress Level"] = stress_level
    if "BMI Category" in FEATURE_COLS:
        d["BMI Category"] = bmi_category
    if "Heart Rate" in FEATURE_COLS:
        d["Heart Rate"] = heart_rate
    if "Daily Steps" in FEATURE_COLS:
        d["Daily Steps"] = daily_steps
    if "Sleep Disorder" in FEATURE_COLS:
        d["Sleep Disorder"] = sleep_disorder
    if "Blood Pressure" in FEATURE_COLS:
        d["Blood Pressure"] = blood_pressure
    for c in FEATURE_COLS:
        if c not in d:
            d[c] = df[c].iloc[0]
    return d

perfil_estresado = build_profile("Male", 35, "Software Engineer", 20, 9, "Obese", 85, 2000, "Insomnia")
perfil_activo = build_profile("Female", 28, "Teacher", 70, 3, "Normal", 68, 12000, "None")
perfil_intermedio = build_profile("Male", 40, "Nurse", 45, 5, "Normal Weight", 75, 6000, "None")
perfiles = [
    ("Persona estresada + poca actividad", perfil_estresado),
    ("Persona activa + estrés bajo", perfil_activo),
    ("Intermedio", perfil_intermedio),
]

demo_results = []
for nombre, perfil in perfiles:
    row = pd.DataFrame([perfil])[FEATURE_COLS]
    pred = best_pipeline.predict(row)[0]
    demo_results.append({"Perfil": nombre, "Predicción (horas)": round(pred, 2)})
demo_df = pd.DataFrame(demo_results)
print(demo_df)

# **Disclaimer:** Esto no es consejo médico; es un ejemplo de cómo los datos encuentran patrones. Las predicciones son ilustrativas y no sustituyen evaluación profesional.

# ---
## Resumen del vídeo (5 bullets para cierre)

# 1. **Objetivo:** Predecir Sleep Duration a partir de hábitos (estrés, actividad, edad, BMI, etc.) con un dataset real.
# 2. **Flujo:** Carga → limpieza mínima → EDA con pocas gráficas → pipeline (ColumnTransformer) → comparativa de modelos (Ridge, RF, HistGradientBoosting/XGBoost).
# 3. **Mejor modelo:** Elegido por MAE en test; interpretación por feature importance o coeficientes.
# 4. **Demo:** Tres perfiles ficticios pasados por el pipeline para mostrar predicciones.
# 5. **Cierre:** Los datos muestran patrones (estrés ↓ sueño, actividad ↑ sueño); no es consejo médico, es ciencia de datos aplicada.
