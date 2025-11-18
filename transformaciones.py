import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

# ==========================
# 1. RUTAS Y CARGA DE DATOS
# ==========================

DATA_PATH = Path("muestra4s.csv")

# Carpeta de salida
TRANSFORM_DIR = Path("Transformacion Estadistica") / "datos_transformados"
GRAF_DIR = TRANSFORM_DIR / "graficas"

TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
GRAF_DIR.mkdir(parents=True, exist_ok=True)

print(f"Carpeta de salida: {TRANSFORM_DIR.resolve()}")

# Cargar datos
df = pd.read_csv(DATA_PATH)

# ==========================
# 2. SELECCIÓN DE VARIABLES
# ==========================

sensor_cols = ["sensor1", "sensor2", "sensor3", "sensor4"]
X = df[sensor_cols]

# ==========================
# 3. STANDARDIZATION (Z-SCORE)
# ==========================

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

df_std = pd.DataFrame(X_std, columns=sensor_cols)
df_std.to_csv(TRANSFORM_DIR / "sensores_estandarizados.csv", index=False)

print("Datos estandarizados guardados.")

# Gráfica: histogramas estandarizados
plt.figure(figsize=(10, 8))
df_std.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.savefig(GRAF_DIR / "histogramas_estandarizados.png", dpi=300)
plt.close()

# ==========================
# 4. NORMALIZACIÓN 0–1
# ==========================

scaler_norm = MinMaxScaler()
X_norm = scaler_norm.fit_transform(X)

df_norm = pd.DataFrame(X_norm, columns=sensor_cols)
df_norm.to_csv(TRANSFORM_DIR / "sensores_normalizados.csv", index=False)

print("Datos normalizados guardados.")

# Gráfica normalizados
plt.figure(figsize=(10, 8))
df_norm.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.savefig(GRAF_DIR / "histogramas_normalizados.png", dpi=300)
plt.close()

# ==========================
# 5. REDUCCIÓN DE RUIDO (Isolation Forest)
# ==========================

iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_flag"] = iso.fit_predict(X)

# 1 = normal, -1 = posible ruido
df_clean = df[df["anomaly_flag"] == 1].reset_index(drop=True)

df_clean.to_csv(TRANSFORM_DIR / "datos_sin_ruido_IF.csv", index=False)

print("Datos reducidos por ruido guardados (Isolation Forest).")

# ==========================
# 6. RESUMEN
# ==========================

print("\n=== ARCHIVOS GENERADOS ===")
for file in os.listdir(TRANSFORM_DIR):
    print(" -", file)
