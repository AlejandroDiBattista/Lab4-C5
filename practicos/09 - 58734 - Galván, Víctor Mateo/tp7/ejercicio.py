import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


@st.cache_data
def cargar_datos():
    df = pd.read_csv('ventas.csv')
    return df


class RedNeuronal(nn.Module):
    def __init__(self, num_neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.capa_entrada = nn.Linear(1, num_neuronas_ocultas)
        self.capa_oculta = nn.Linear(num_neuronas_ocultas, 1)

    def forward(self, x):
        x = torch.relu(self.capa_entrada(x))
        x = self.capa_oculta(x)
        return x


st.title("Estimación de Ventas Diarias")


st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.number_input("Aprendizaje", 0.0, 1.0, 0.01)
cant_epocas = st.sidebar.number_input("Repeticiones", 10, 10000, 1000)
neuronas_ocultas = st.sidebar.number_input("Neuronas Capa Oculta", 1, 100, 10)
boton_entrenar = st.sidebar.button("Entrenar")


df = cargar_datos()
x = df['día'].values.reshape(-1, 1)
y = df['ventas'].values.reshape(-1, 1)

# Normalizar los datos
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)


x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)


dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def entrenar_red_neuronal():
    modelo = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    historial_costo = []
    
    progreso = st.sidebar.progress(0)
    estado_texto = st.sidebar.empty()
    
    for epoca in range(cant_epocas):
        for datos in dataloader:
            entradas, salidas = datos
            optimizador.zero_grad()
            predicciones = modelo(entradas)
            perdida = criterio(predicciones, salidas)
            perdida.backward()
            optimizador.step()
        
        historial_costo.append(perdida.item())
        
        progreso.progress((epoca + 1) / cant_epocas)
        estado_texto.text(f"Epoch {epoca + 1}/{cant_epocas} - Error: {perdida.item():.6f}")
    
    progreso.empty()
    estado_texto.text("Entrenamiento exitoso")
    return modelo, historial_costo


if boton_entrenar:
    modelo, historial_costo = entrenar_red_neuronal()
    
   
    st.subheader("Evolución de la función de costo")
    fig, ax = plt.subplots()
    ax.plot(range(cant_epocas), historial_costo, color="green", label="Pérdidas")
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.legend()
    st.pyplot(fig)

    
    modelo.eval()
    predicciones = modelo(x_tensor).detach().numpy()
    predicciones_ventas = scaler_y.inverse_transform(predicciones)
    
   
    st.subheader("Estimación de Ventas Diarias")
    fig, ax = plt.subplots()
    ax.scatter(df['día'], df['ventas'], color='blue', label='Datos Reales')
    ax.plot(df['día'], predicciones_ventas, color='red', label='Curva de Ajuste')
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
