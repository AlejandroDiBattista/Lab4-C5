import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Configuración principal
st.set_page_config(page_title="Predicción de Ventas", layout="wide")
st.title("Estimación de Ventas Diarias")

# Clase para la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, entrada, ocultas, salida):
        super(RedNeuronal, self).__init__()
        self.capa_oculta = nn.Linear(entrada, ocultas)
        self.capa_salida = nn.Linear(ocultas, salida)

    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))
        return self.capa_salida(x)

# Funciones de normalización y desnormalización
def normalizar(datos):
    return (datos - np.min(datos)) / (np.max(datos) - np.min(datos))

def desnormalizar(datos_normalizados, datos_originales):
    return datos_normalizados * (np.max(datos_originales) - np.min(datos_originales)) + np.min(datos_originales)

# Cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("ventas.csv")
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo 'ventas.csv'.")
        return pd.DataFrame(columns=["dia", "ventas"])

# Verificar y cargar datos
datos = cargar_datos()
if datos.empty:
    st.stop()

if 'dia' not in datos.columns or 'ventas' not in datos.columns:
    st.error("Las columnas 'dia' y 'ventas' deben estar presentes en el archivo CSV.")
    st.stop()

# Preparar los datos
dias = datos['dia'].values.reshape(-1, 1)
ventas = datos['ventas'].values.reshape(-1, 1)
dias_normalizados = normalizar(dias)
ventas_normalizadas = normalizar(ventas)

dias_tensor = torch.FloatTensor(dias_normalizados)
ventas_tensor = torch.FloatTensor(ventas_normalizadas)

# Configuración en la barra lateral
st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.number_input("Aprendizaje", 0.001, 1.0, 0.01, step=0.01, format="%.3f")
epocas = st.sidebar.number_input("Repeticiones", 10, 10000, 1000, step=10)
neuronas_ocultas = st.sidebar.number_input("Neuronas Capa Oculta", 1, 100, 10)

# Crear modelo
modelo = RedNeuronal(1, neuronas_ocultas, 1)
funcion_perdida = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

# Entrenamiento
if st.sidebar.button("Entrenar"):
    st.write("### Estado del Entrenamiento")
    barra_progreso = st.progress(0)
    estado_entrenamiento = st.empty()
    estado_entrenamiento.info("Entrenando...")
    historial_perdidas = []

    for epoca in range(epocas):
        modelo.train()
        optimizador.zero_grad()

        predicciones = modelo(dias_tensor)
        perdida = funcion_perdida(predicciones, ventas_tensor)

        perdida.backward()
        optimizador.step()

        historial_perdidas.append(perdida.item())
        barra_progreso.progress((epoca + 1) / epocas)

    estado_entrenamiento.success("Entrenamiento exitoso")
    st.write(f"Epoch {epocas}/{epocas} - Error: {historial_perdidas[-1]:.6f}")

    # Gráfico de pérdida
    st.write("### Evolución de la Pérdida")
    fig_perdida, ax_perdida = plt.subplots(figsize=(7, 4))
    ax_perdida.plot(historial_perdidas, label="Pérdidas", color="green")
    ax_perdida.set_xlabel("Época")
    ax_perdida.set_ylabel("Pérdida")
    ax_perdida.set_title("Evolución de la Pérdida")
    ax_perdida.legend()
    st.pyplot(fig_perdida)

    # Predicción
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(dias_tensor).numpy()
        predicciones_desnormalizadas = desnormalizar(predicciones, ventas)

    # Gráfico de predicciones
    st.write("### Estimación de Ventas Diarias")
    fig_prediccion, ax_prediccion = plt.subplots(figsize=(7, 5))
    ax_prediccion.scatter(datos['dia'], datos['ventas'], color='blue', label="Datos Reales")
    ax_prediccion.plot(datos['dia'], predicciones_desnormalizadas, color='red', label="Curva de Ajuste")
    ax_prediccion.set_xlabel("Día del Mes")
    ax_prediccion.set_ylabel("Ventas")
    ax_prediccion.set_title("Estimación de Ventas Diarias")
    ax_prediccion.legend()
    st.pyplot(fig_prediccion)
