import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Crear Red Neuronal
class RedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activacion = nn.ReLU()

    def forward(self, x):
        x = self.activacion(self.hidden(x))
        x = self.output(x)
        return x
    
## Leer Datos
def leer_datos():
    datos = pd.read_csv("ventas.csv")
    return datos

datos = leer_datos()

## Normalizar Datos
def normalizar_datos(datos):
    x = datos["dia"].values.reshape(-1, 1)
    y = datos["ventas"].values.reshape(-1, 1)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return x, y, x_norm, y_norm, x_min, x_max, y_min, y_max

## Entrenar Red Neuronal
def entrenar_red(modelo, datos, num_epochs, learning_rate):
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    perdida_historial = []

    barra_progreso.progress(0)

    for epoch in range(num_epochs):
        predicciones = modelo(datos["x_tensor"])
        perdida = criterio(predicciones, datos["y_tensor"])
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        perdida_historial.append(perdida.item())
        barra_progreso.progress((epoch + 1) / num_epochs)

        progreso = f"Epoch {num_epochs}/{num_epochs} - Error: {perdida.item():.6f}"
        st.sidebar.text(progreso)

        return perdida_historial

## Guardar Modelo
def guardar_modelo(modelo, path="modelo_entrenado.pth"):
    torch.save(modelo.state_dict(), path)

## Graficar Predicciones
def graficar_predicciones(x, y, x_plot, y_pred_rescaled):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Datos Reales")
    ax.plot(x_plot, y_pred_rescaled, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Dia del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

st.title('Estimación de Ventas Diarias')

st.sidebar.title("Parametros de Entrenamiento")
learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
num_epochs = st.sidebar.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=10)
hidden_size = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10, step=1)

if st.sidebar.button("Entrenar"):
    datos = leer_datos()
    x, y, x_norm, y_norm, x_min, x_max, y_min, y_max = normalizar_datos(datos)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    datos_tensor = {"x_tensor": x_tensor, "y_tensor": y_tensor}

    modelo = RedNeuronal(1, hidden_size, 1)
    barra_progreso = st.sidebar.progress(0)
    perdida_historial = entrenar_red(modelo, datos_tensor, num_epochs, learning_rate)

    st.sidebar.success("Entrenamiento Exitoso")

    fig, ax = plt.subplots()
    ax.plot(perdida_historial, label="")
    ax.set_xlabel("Època")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.sidebar.pyplot(fig)

    x_plot = np.linspace(1, 31, 100).reshape(-1, 1)
    x_plot_norm = (x_plot -x_min) / (x_max - x_min)
    x_plot_tensor = torch.tensor(x_plot_norm, dtype=torch.float32)
    with torch.no_grad():
        y_pred = modelo(x_plot_tensor).numpy()
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min

    graficar_predicciones(x, y, x_plot, y_pred_rescaled)


