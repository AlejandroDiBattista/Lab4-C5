import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Crear Red Neuronal
class Net(nn.Module):
    def __init__(self, cantidad_de_neuronas):
        super(Net, self).__init__()
        self.c1 = nn.Linear(1, 1)
        self.c2 = nn.Linear(1, cantidad_de_neuronas)
        self.c3 = nn.Linear(cantidad_de_neuronas, 1)
        self.activacion_relu = nn.ReLU()
    
    def forward(self, x):
        x = self.activacion_relu(self.c1(x))
        x = self.activacion_relu(self.c2(x))
        x = self.c3(x)
        return x

# Leer Datos
datos = pd.read_csv("ventas.csv")

# Normalizar Datos
dia_min = np.min(datos['dia'])
dia_max = np.max(datos['dia'])
x = (datos['dia'] - dia_min) / (dia_max - dia_min)

ventas_min = np.min(datos['ventas'])
ventas_max = np.max(datos['ventas'])
y = (datos['ventas'] - ventas_min) / (ventas_max - ventas_min)

x_tensor = torch.tensor(x.values, dtype=torch.float32).view(-1,1)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

# Entrenar Red neuronal
st.title('Estimación de Ventas Diarias')
st.sidebar.title("Parámetros de Entrenamiento")

col1, col2 = st.sidebar.columns(2)

with col1:
    tasa_aprendisaje = st.number_input("Tasa de aprendizaje", min_value=0.0, max_value=1.0, value=0.1)

with col2:
    epocas = st.number_input("Épocas", min_value=10, max_value=10000, value=100)

cantidad_de_neuronas = st.sidebar.number_input("Neuronas en capa oculta", min_value=1, max_value=100, value=5)

if st.sidebar.button("Entrenar"):
    red = Net(cantidad_de_neuronas)
    
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(red.parameters(), lr=tasa_aprendisaje)

    barra_progreso_sidebar = st.sidebar.progress(0)
    perdidas = []
    
    for epoca in range(epocas):
        predicciones = red(x_tensor)
        loss = criterio(predicciones, y_tensor)
        perdidas.append(loss.item())
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()
        barra_progreso_sidebar.progress((epoca + 1) / epocas)
        
    st.sidebar.write(f"Épocas {epoca + 1}/{epocas} - Error: {loss.item():.5f}")
    st.sidebar.success("Entrenamiento exitoso.")
    
    predicciones_desnormalizadas = predicciones.detach().numpy() * (ventas_max - ventas_min) + ventas_min
    ventas_reales_desnormalizadas = y * (ventas_max - ventas_min) + ventas_min

    # Graficar las predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(datos["dia"], ventas_reales_desnormalizadas, label="Datos reales", color="blue")
    plt.plot(datos["dia"], predicciones_desnormalizadas, label='Curva de ajuste', linestyle='-', color="red")
    plt.xlabel("Días del mes")
    plt.ylabel("Ventas")
    plt.legend()
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    plt.plot(range(epocas), perdidas, label="Pérdida", color="green")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    st.sidebar.pyplot(plt)

    torch.save(red.state_dict(), "red_entrenada.pth")