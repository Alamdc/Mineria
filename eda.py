import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. RUTAS Y CARGA DE DATOS

# Ruta del archivo CSV 
DATA_PATH = Path("muestra4s.csv")

# Carpeta donde se guardarán las gráficas
OUTPUT_DIR = Path("resultados_mineria") / "graficas"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  

print(f"Archivo de datos: {DATA_PATH.resolve()}")
print(f"Carpeta de salida: {OUTPUT_DIR.resolve()}")

# Cargar datos
df = pd.read_csv(DATA_PATH)

# 2. INFO GENERAL DEL DATASET

print("\n=== Shape del DataFrame ===")
print(df.shape)

print("\n=== Primeras filas ===")
print(df.head())

print("\n=== Info ===")
df.info()

print("\n=== Descriptivos ===")
print(df.describe())

print("\n=== Valores nulos por columna ===")
print(df.isna().sum())

# guardar resumen descriptivo a CSV
df.describe().to_csv(OUTPUT_DIR / "resumen_descriptivo.csv", index=True)

# 3. SELECCIÓN DE VARIABLES NUMÉRICAS

# Quitamos 'id' para los análisis de sensores
sensor_cols = ["sensor1", "sensor2", "sensor3", "sensor4"]
num_df = df[sensor_cols]

# 4. HISTOGRAMAS POR SENSOR

plt.figure(figsize=(10, 8))
num_df.hist(bins=20, figsize=(10, 8))
plt.tight_layout()

hist_path = OUTPUT_DIR / "histogramas_sensores.png"
plt.savefig(hist_path, dpi=300)
plt.close()

print(f"Histograma guardado en: {hist_path}")

# 5. MATRIZ DE CORRELACIÓN

corr = num_df.corr()
print("\n=== Matriz de correlación ===")
print(corr)

plt.figure(figsize=(6, 5))
plt.imshow(corr, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Matriz de correlación - Sensores")
plt.tight_layout()

corr_path = OUTPUT_DIR / "matriz_correlacion_sensores.png"
plt.savefig(corr_path, dpi=300)
plt.close()

print(f"Matriz de correlación guardada en: {corr_path}")

