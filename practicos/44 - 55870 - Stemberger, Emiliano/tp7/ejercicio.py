import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Verificar si el archivo de datos existe
ruta_archivo = "ventas.csv"  # Ajustar la ruta si es necesario

if os.path.exists(ruta_archivo):
    st.write("Archivo 'ventas.csv' encontrado correctamente.")
else:
    st.error("No se encuentra el archivo 'ventas.csv'. Verifica la ruta e inténtalo de nuevo.")
    st.stop()

# Definición de la Red Neuronal
class RedNeuronal(nn.Module):
    def __init__(self, num_neuronas):
        super(RedNeuronal, self).__init__()
        self.capa_oculta = nn.Linear(1, num_neuronas)
        self.capa_salida = nn.Linear(num_neuronas, 1)

    def forward(self, entrada):
        activacion = torch.relu(self.capa_oculta(entrada))
        salida = self.capa_salida(activacion)
        return salida

# Función para cargar los datos
def leer_datos():
    datos_df = pd.read_csv(ruta_archivo)
    return datos_df

datos = leer_datos()

# Separar las columnas de entrada y salida
x_entrada = datos[['dia']].values
y_salida = datos[['ventas']].values

# Normalizar los datos
x_min, x_max = x_entrada.min(), x_entrada.max()
y_min, y_max = y_salida.min(), y_salida.max()

x_normalizado = (x_entrada - x_min) / (x_max - x_min)
y_normalizado = (y_salida - y_min) / (y_max - y_min)

# Convertir a tensores de PyTorch
x_tensor = torch.tensor(x_normalizado, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y_normalizado, dtype=torch.float32).reshape(-1, 1)

# Función para entrenar la red
def entrenar_red(tasa_aprendizaje, epocas, num_neuronas):
    red = RedNeuronal(num_neuronas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(red.parameters(), lr=tasa_aprendizaje)

    barra_progreso = st.progress(0)
    historial_perdida = []

    for epoca in range(epocas):
        red.train()
        optimizador.zero_grad()
        predicciones = red(x_tensor)
        perdida = criterio(predicciones, y_tensor)
        perdida.backward()
        optimizador.step()

        historial_perdida.append(perdida.item())
        barra_progreso.progress((epoca + 1) / epocas)

    return red, historial_perdida

# Función para guardar el modelo
def guardar_red(red, nombre_archivo="modelo_ventas.pth"):
    torch.save(red.state_dict(), nombre_archivo)
    st.write(f"Modelo guardado en {nombre_archivo}")

# Función opcional para cargar un modelo existente
def cargar_red(nombre_archivo="modelo_ventas.pth", num_neuronas=5):
    red = RedNeuronal(num_neuronas)
    red.load_state_dict(torch.load(nombre_archivo))
    red.eval()
    return red

# Graficar resultados
def graficar_resultados(x, y_real, y_estimado, historial_perdida):
    figura, (grafico_perdida, grafico_prediccion) = plt.subplots(2, 1, figsize=(8, 10))

    # Gráfico de la pérdida
    grafico_perdida.plot(historial_perdida, label="Pérdida")
    grafico_perdida.set_xlabel("Épocas")
    grafico_perdida.set_ylabel("Pérdida")
    grafico_perdida.legend()

    # Gráfico de las predicciones
    grafico_prediccion.scatter(x, y_real, color='blue', label="Datos reales")
    grafico_prediccion.plot(x, y_estimado, color='red', label="Predicción")
    grafico_prediccion.set_xlabel("Día")
    grafico_prediccion.set_ylabel("Ventas")
    grafico_prediccion.legend()

    st.pyplot(figura)

# Interfaz de Streamlit
st.title("Modelo de Estimación de Ventas")

# Configuración de parámetros mediante controles
tasa_aprendizaje = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 1.0, 0.1)
num_epocas = st.sidebar.slider("Número de Épocas", 10, 10000, 500)
num_neuronas = st.sidebar.slider("Neurona en Capa Oculta", 1, 100, 10)

if st.sidebar.button("Iniciar Entrenamiento"):
    red, historial_perdida = entrenar_red(tasa_aprendizaje, num_epocas, num_neuronas)

    # Generar predicciones
    with torch.no_grad():
        predicciones_normalizadas = red(x_tensor).numpy()

    # Desnormalizar para graficar
    predicciones = predicciones_normalizadas * (y_max - y_min) + y_min

    # Mostrar resultados
    st.success("Entrenamiento completado con éxito.")
    graficar_resultados(x_entrada, y_salida, predicciones, historial_perdida)
