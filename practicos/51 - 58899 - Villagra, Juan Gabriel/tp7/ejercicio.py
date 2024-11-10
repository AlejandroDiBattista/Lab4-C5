import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class RedVentas(nn.Module):
    def __init__(self, neuronas_escondidas):
        super(RedVentas, self).__init__()
        self.capa1 = nn.Linear(1, neuronas_escondidas)
        self.relu = nn.ReLU()
        self.capa2 = nn.Linear(neuronas_escondidas, 1)
        
    def forward(self, x):
        x = self.capa1(x)
        x = self.relu(x)
        x = self.capa2(x)
        return x

class DatosVentas(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def entrenar_red(red, datos, funcion_error, optimizador, epocas, barra):
    errores = []
    for e in range(epocas):
        for x, y in datos:
            optimizador.zero_grad()
            y_pred = red(x)
            error = funcion_error(y_pred, y)
            error.backward()
            optimizador.step()
        
        errores.append(error.item())
        barra.progress((e + 1) / epocas)
        
    return errores

st.title('Predicción de Ventas con Red Neuronal')

st.sidebar.header('Configuración')

velocidad_aprendizaje = st.sidebar.number_input(
    'Velocidad de Aprendizaje',
    min_value=0.0,
    max_value=1.0,
    value=0.01,
    step=0.001,
    format="%.3f"
)

num_epocas = st.sidebar.number_input(
    'Número de Épocas',
    min_value=10,
    max_value=10000,
    value=500,
    step=10
)

neuronas_escondidas = st.sidebar.number_input(
    'Neuronas Escondidas',
    min_value=1,
    max_value=100,
    value=15,
    step=1
)

@st.cache_data
def cargar_datos():
    df = pd.read_csv('ventas.csv')
    X = df['dia'].values.reshape(-1, 1)
    y = df['ventas'].values.reshape(-1, 1)
    return X, y, df

X, y, df = cargar_datos()

datos = DatosVentas(X, y)
cargador = DataLoader(datos, batch_size=len(datos), shuffle=True)

red = RedVentas(neuronas_escondidas)
funcion_error = nn.MSELoss()
optimizador = torch.optim.Adam(red.parameters(), lr=velocidad_aprendizaje)

if st.sidebar.button('Entrenar Red'):
    barra = st.sidebar.progress(0)
    
    errores = entrenar_red(red, cargador, funcion_error, optimizador, num_epocas, barra)
    
    st.sidebar.success('¡Entrenamiento listo!')
    
    fig_error, ax_error = plt.subplots()
    ax_error.plot(errores, label='Error', color='red')
    ax_error.set_xlabel('Época')
    ax_error.set_ylabel('Error')
    ax_error.legend()
    st.sidebar.pyplot(fig_error)
    
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        predicciones = red(X_tensor).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Datos Reales')
    ax.plot(X, predicciones, color='red', label='Predicción')
    ax.set_xlabel('Día')
    ax.set_ylabel('Ventas')
    ax.set_title('Predicción de Ventas Diarias')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Datos Reales')
    ax.set_xlabel('Día')
    ax.set_ylabel('Ventas')
    ax.set_title('Datos de Ventas')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

color_datos = st.sidebar.color_picker('Color de Datos Reales', '#1E90FF')
color_prediccion = st.sidebar.color_picker('Color de Predicción', '#FF4500')

if 'red' in locals():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color=color_datos, label='Datos Reales')
    ax.plot(X, predicciones, color=color_prediccion, label='Predicción')
    ax.set_xlabel('Día')
    ax.set_ylabel('Ventas')
    ax.set_title('Predicción de Ventas Diarias')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)