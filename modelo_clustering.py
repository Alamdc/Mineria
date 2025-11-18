import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================================
# 1. RUTAS Y CARGA DE DATOS TRANSFORMADOS (ESTANDARIZADOS)
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "Transformacion Estadistica" / "datos_transformados" / "sensores_estandarizados.csv"
df = pd.read_csv(data_path)

sensor_cols = ["sensor1", "sensor2", "sensor3", "sensor4"]
X = df[sensor_cols]

# Carpeta de resultados
OUTPUT_DIR = Path("aprendizaje_Automatico") / "modelos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. PCA - REDUCCIÓN A 2 COMPONENTES
# ==========================================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca.to_csv(OUTPUT_DIR / "pca_componentes.csv", index=False)

print("PCA realizado. Componentes guardados.")

# ==========================================================
# 3. K-MEANS PARA DIFERENTES k
# ==========================================================

k_values = [3, 4, 5, 15]
resultados = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    silhouette = silhouette_score(X_pca, labels)
    resultados.append([k, silhouette])

    # Guardar clusters
    df_plot = df_pca.copy()
    df_plot["cluster"] = labels
    df_plot.to_csv(OUTPUT_DIR / f"kmeans_k{k}.csv", index=False)

    # Gráfica PCA con clusters
    plt.figure(figsize=(7,5))
    plt.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["cluster"], cmap="viridis")
    plt.title(f"K-Means con k={k} (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"pca_kmeans_k{k}.png", dpi=300)
    plt.close()

print("K-Means finalizado. Clusters y gráficas guardadas.")

# ==========================================================
# 4. GUARDAR TABLA DE SILHOUETTE
# ==========================================================

df_resultados = pd.DataFrame(resultados, columns=["k", "silhouette"])
df_resultados.to_csv(OUTPUT_DIR / "silhouette_scores.csv", index=False)

print("\n=== Resultados Silhouette ===")
print(df_resultados)
