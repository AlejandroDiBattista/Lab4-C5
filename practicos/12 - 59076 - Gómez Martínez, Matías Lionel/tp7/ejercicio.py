import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class ModeloVentas(nn.Module):
    def __init__(self, entrada, ocultas, salida):
        super().__init__()
        self.oculta = nn.Linear(entrada, ocultas)
        self.salida = nn.Linear(ocultas, salida)
        self.activacion = nn.ReLU()

    def forward(self, x):
        x = self.activacion(self.oculta(x))
        return self.salida(x)

def cargar_datos(ruta="ventas.csv"):
    return pd.read_csv(ruta)

def normalizar(x, y):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return x_norm, y_norm, x_min, x_max, y_min, y_max

def entrenamiento(modelo, datos, epocas, lr):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.MSELoss()
    historial_perdidas = []

    for epoca in range(epocas):
        pred = modelo(datos["x"])
        perdida = criterio(pred, datos["y"])
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        historial_perdidas.append(perdida.item())
        barra_progreso.progress((epoca + 1) / epocas)

    return historial_perdidas

def mostrar_predicciones(x_original, y_original, x_pred, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(x_original, y_original, color="blue", label="Datos Reales")
    ax.plot(x_pred, y_pred, color="red", label="Predicción")
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

def guardar(modelo, nombre_archivo="modelo_ventas.pth"):
    torch.save(modelo.state_dict(), nombre_archivo)

st.title("Predicción de Ventas Diarias")
st.sidebar.header("Parámetros de Entrenamiento")

with st.sidebar:
    col1, col2 = st.columns(2)

    tasa_aprendizaje = col1.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    epocas = col2.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=10)
    neuronas_ocultas = st.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10, step=1)

if st.sidebar.button("Iniciar Entrenamiento"):
    datos = cargar_datos()
    x, y = datos["dia"].values.reshape(-1, 1), datos["ventas"].values.reshape(-1, 1)
    x_norm, y_norm, x_min, x_max, y_min, y_max = normalizar(x, y)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    datos_tensor = {"x": x_tensor, "y": y_tensor}

    red = ModeloVentas(entrada=1, ocultas=neuronas_ocultas, salida=1)
    barra_progreso = st.sidebar.progress(0)
    historial = entrenamiento(red, datos_tensor, epocas, tasa_aprendizaje)

    st.sidebar.success("Entrenamiento completado")
    fig, ax = plt.subplots()
    ax.plot(historial, label="Pérdida")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.sidebar.pyplot(fig)

    x_pred = np.linspace(1, 31, 100).reshape(-1, 1)
    x_pred_norm = (x_pred - x_min) / (x_max - x_min)
    x_pred_tensor = torch.tensor(x_pred_norm, dtype=torch.float32)
    with torch.no_grad():
        y_pred_norm = red(x_pred_tensor).numpy()
    y_pred = y_pred_norm * (y_max - y_min) + y_min

    mostrar_predicciones(x, y, x_pred, y_pred)
