import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

os.environ["OMP_NUM_THREADS"] = "4"

st.sidebar.title("Clustering y Análisis de Datos de Pacientes")
st.sidebar.header("Cargar Datos")

# Paso 1: subir el archivo CSV
file_path = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'])

if file_path is not None:
    data = pd.read_csv(file_path)
    st.write("Primeras filas del dataset:\n", data.head())

    # Paso 2: Limpieza de datos
    data['Enfermedad'] = data['Enfermedad'].map({'NO': 0, 'SI': 1})
    st.write(f"Número de filas duplicadas: {data.duplicated().sum()}")

    data_cleaned = data.drop_duplicates()
    st.write("Valores nulos en cada columna:\n", data_cleaned.isnull().sum())

    st.write("Información del dataset después de la limpieza:")
    buffer = StringIO()
    data_cleaned.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Visualización antes de la normalización
    if st.sidebar.checkbox("Mostrar visualización antes de la normalización"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data_cleaned.drop(columns=['NOEXPED']))
        plt.title('Visualización antes de la Estandarización')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Paso 3: Preparación de los datos
    data_features = data_cleaned.drop(columns=['NOEXPED'])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    # Visualización después de la normalización
    if st.sidebar.checkbox("Mostrar visualización después de la normalización"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pd.DataFrame(data_scaled, columns=data_features.columns))
        plt.title('Visualización después de la Estandarización')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Paso 4: Aplicación de PCA para reducción de dimensiones
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    st.write(f"Varianza explicada por los 2 componentes principales: {pca.explained_variance_ratio_}")

    # Paso 5: Aplicar KMeans para hacer clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_pca)
    st.write(f"Centroides de los 3 clusters:\n{kmeans.cluster_centers_}")

    # Paso 6: Visualización de los clusters con PCA
    if st.sidebar.checkbox("Mostrar visualización de clustering"):
        plt.figure(figsize=(8, 6))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50)
        plt.title('Clustering de Pacientes usando KMeans y PCA')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(label='Cluster')
        st.pyplot(plt)

    # Paso 7: Agregar la información de los clusters al dataset limpio
    data_cleaned['Cluster'] = clusters

    # Paso 8: Selección de gráfico para visualizar la distribución de pacientes por cluster
    option = st.sidebar.selectbox(
        'Selecciona el tipo de gráfico para visualizar la distribución por cluster:',
        ('Distribución de Pacientes por Cluster', 'Distribución de la Edad por Cluster', 
         'Relación entre Hipertensión y Clusters', 'Relación entre Hiperglucemia y Clusters')
    )

    # Visualización según la opción seleccionada
    if option == 'Distribución de Pacientes por Cluster':
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Cluster', data=data_cleaned, palette='viridis')
        plt.title('Distribución de Pacientes por Cluster')
        st.pyplot(plt)

    elif option == 'Distribución de la Edad por Cluster':
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y='EDAD', data=data_cleaned, palette='viridis')
        plt.title('Distribución de la Edad por Cluster')
        st.pyplot(plt)

    elif option == 'Relación entre Hipertensión y Clusters':
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Cluster', hue='HIPERTEN', data=data_cleaned, palette='viridis')
        plt.title('Relación entre Hipertensión y Clusters')
        st.pyplot(plt)

    elif option == 'Relación entre Hiperglucemia y Clusters':
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Cluster', hue='HIPERGLU', data=data_cleaned, palette='viridis')
        plt.title('Relación entre Hiperglucemia y Clusters')
        st.pyplot(plt)
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")

