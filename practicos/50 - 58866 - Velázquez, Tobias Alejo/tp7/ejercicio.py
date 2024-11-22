import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones

st.set_page_config(layout="wide")
st.title("Estimación de Ventas Diarias")

with st.sidebar:
    st.header("Parámetros de Entrenamiento")
    col1, col2 = st.columns(2)
    with col1:
        tasa_aprendizaje = st.number_input("Tasa de aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    with col2:
        epocas = st.number_input("Repeticiones", min_value=10, max_value=10000, value=100, step=10)
    neuronas_ocultas = st.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=5, step=1)
    entrenar = st.button("Entrenar")
    
    barra_progreso = st.progress(0)
    mensaje_exitoso = st.empty()
    grafico_perdida = st.empty()

@st.cache_data
def cargar_datos():
    return pd.read_csv("ventas.csv")

datos = cargar_datos()

def normalizar(datos):
    max_val = datos.max()
    min_val = datos.min()
    return (datos - min_val) / (max_val - min_val), max_val, min_val

ventas_norm, max_ventas, min_ventas = normalizar(datos['ventas'])

class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(1, neuronas_ocultas)
        self.fc2 = nn.Linear(neuronas_ocultas, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

x = torch.tensor(datos['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(ventas_norm.values, dtype=torch.float32).view(-1, 1)

if entrenar:
    red = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(red.parameters(), lr=tasa_aprendizaje)
    
    perdidas = []
    for epoch in range(epocas):
        predicciones = red(x)
        perdida = criterio(predicciones, y)
        perdidas.append(perdida.item())

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        barra_progreso.progress((epoch + 1) / epocas)
    
    mensaje_exitoso.success("Entrenamiento exitoso")

    fig_costo, ax_costo = plt.subplots(figsize=(3, 2))
    ax_costo.plot(range(epocas), perdidas, color="green", label="Pérdidas")
    ax_costo.set_xlabel("Época")
    ax_costo.set_ylabel("Pérdida")
    ax_costo.set_title("Reducción de la Función de Costo")
    ax_costo.legend()
    grafico_perdida.pyplot(fig_costo)

    x_test = torch.linspace(1, 30, 100).view(-1, 1)
    predicciones = red(x_test).detach().numpy()
    predicciones_desnormalizadas = predicciones * (max_ventas - min_ventas) + min_ventas

    col2 = st.container()
    with col2:
        st.subheader("Estimación de Ventas Diarias")
        fig_resultados, ax_resultados = plt.subplots(figsize=(6, 4))
        ax_resultados.scatter(datos['dia'], datos['ventas'], label="Datos Reales", color="blue")
        ax_resultados.plot(x_test.numpy(), predicciones_desnormalizadas, label="Curva de Ajuste", color="red")
        ax_resultados.set_xlabel("Día del Mes")
        ax_resultados.set_ylabel("Ventas")
        ax_resultados.set_title("Estimación de Ventas Diarias")
        ax_resultados.legend()
        st.pyplot(fig_resultados)
