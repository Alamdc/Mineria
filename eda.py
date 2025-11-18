import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Importamos la librería 'os' para manejo de carpetas

# --- 1. Definir la carpeta de salida ---
output_folder = "graficas_eda"
os.makedirs(output_folder, exist_ok=True)
print(f"Los gráficos se guardarán en la carpeta: '{output_folder}'")

# Set plot style
sns.set(style="whitegrid")

# Cargar el dataset
try:
    df = pd.read_csv("muestra4s.csv")

    # --- 2. Inspección Inicial ---
    print("\n--- Información General del DataFrame ---")
    df.info()
    print("\n--- Primeras 5 Filas del DataFrame ---")
    print(df.head())

    # Definir columnas de características (excluyendo 'id')
    if 'id' in df.columns:
        features = df.columns.drop('id')
        data_for_analysis = df[features]
    else:
        features = df.columns
        data_for_analysis = df.copy()
        print("\nAdvertencia: No se encontró la columna 'id'. Se usarán todas las columnas como características.")

    # --- 3. Estadísticas Descriptivas ---
    print("\n--- Estadísticas Descriptivas de los Sensores ---")
    print(data_for_analysis.describe())

    # --- 4. Calidad de Datos (Nulos y Duplicados) ---
    print("\n--- Conteo de Valores Nulos por Columna ---")
    print(data_for_analysis.isnull().sum())
    print(f"\n--- Conteo de Filas Duplicadas (solo en sensores) ---")
    print(f"Número de filas duplicadas: {data_for_analysis.duplicated().sum()}")

    # --- 5. Visualizaciones (guardando en la carpeta) ---

    # Histogramas
    print("\nGenerando histogramas...")
    plt.figure(figsize=(12, 8))
    data_for_analysis.hist(bins=30, layout=(2, 2), figsize=(12, 8))
    plt.suptitle('Histogramas de Distribución de los Sensores', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sensor_histograms.png'))
    plt.clf() 

    # Mapa de Calor de Correlación
    print("Generando mapa de calor de correlación...")
    corr_matrix = data_for_analysis.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Mapa de Calor de Correlación entre Sensores')
    plt.savefig(os.path.join(output_folder, 'sensor_correlation_heatmap.png'))
    plt.clf()

    # Pair Plot (Gráfico de Pares)
    print("Generando pair plot...")
    pairplot_fig = sns.pairplot(data_for_analysis)
    pairplot_fig.fig.suptitle('Pair Plot de los Sensores', y=1.02)
    plt.savefig(os.path.join(output_folder, 'sensor_pairplot.png'))
    plt.clf() 

    print(f"\n--- EDA completado. Se generaron 3 gráficos en la carpeta '{output_folder}': ---")
    print(f"1. {os.path.join(output_folder, 'sensor_histograms.png')}")
    print(f"2. {os.path.join(output_folder, 'sensor_correlation_heatmap.png')}")
    print(f"3. {os.path.join(output_folder, 'sensor_pairplot.png')}")

except FileNotFoundError:
    print("Error: El archivo 'muestra4s.csv' no se encontró.")
except Exception as e:
    print(f"Ocurrió un error: {e}")