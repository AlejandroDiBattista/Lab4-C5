import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones

st.title('Estimación de Ventas Diarias')

st.sidebar.header('Parámetros de Entrenamiento')
tasa_de_aprendizaje = st.sidebar.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1)
epocas = st.sidebar.slider('Cantidad de Épocas', 10, 10000, 100)
neuronas_ocultas = st.sidebar.slider('Neuronas en Capa Oculta', 1, 100, 5)
boton_entrenar = st.sidebar.button('Entrenar')


@st.cache_data
def cargar_datos():
    datos = pd.read_csv('ventas.csv')
    return datos

datos = cargar_datos()

x_datos = datos[['dia']].values
y_datos = datos[['ventas']].values

min_x, max_x = x_datos.min(), x_datos.max()
min_y, max_y = y_datos.min(), y_datos.max()

x_datos_normalizado = (x_datos - min_x) / (max_x - min_x)
y_datos_normalizado = (y_datos - min_y) / (max_y - min_y)

class RedNeuronal(nn.Module):
    def __init__(self, entradas, ocultas, salidas):
        super(RedNeuronal, self).__init__()
        self.capa_oculta = nn.Linear(entradas, ocultas)
        self.relu = nn.ReLU()
        self.capa_salida = nn.Linear(ocultas, salidas)
    
    def forward(self, x):
        x = self.capa_oculta(x)
        x = self.relu(x)
        x = self.capa_salida(x)
        return x
    

def entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, tasa_aprendizaje, epocas):
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    historial_perdida = []

    for epoca in range(epocas):
        modelo.train()
    
        entradas = torch.tensor(x_entrenamiento, dtype=torch.float32)
        objetivos = torch.tensor(y_entrenamiento, dtype=torch.float32)

        salidas = modelo(entradas)
        perdida = criterio(salidas, objetivos)

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
    
        historial_perdida.append(perdida.item())

        if epoca % 10 == 0:
            st.sidebar.progress((epoca + 1) / epocas)

    return modelo, historial_perdida

if boton_entrenar:
    modelo = RedNeuronal(entradas=1, ocultas=neuronas_ocultas, salidas=1)
    modelo, historial_perdida = entrenar_modelo(modelo, x_datos_normalizado, y_datos_normalizado, tasa_de_aprendizaje, epocas)
    
    st.sidebar.success('Entrenamiento exitoso')
    
    
    fig, ax = plt.subplots()
    ax.plot(historial_perdida, color='green', label='Pérdida')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.legend()
    st.sidebar.pyplot(fig)

    
    modelo.eval()
    with torch.no_grad():
        predicciones_normalizadas = modelo(torch.tensor(x_datos_normalizado, dtype=torch.float32)).numpy()
    
        predicciones = predicciones_normalizadas * (max_y - min_y) + min_y
        ventas_reales = datos['ventas'].values

        
        fig, ax = plt.subplots()
        ax.plot(datos['dia'], ventas_reales, 'bo', label='Datos Reales')
        ax.plot(datos['dia'], predicciones, 'r-', label='Curva de Ajuste')
        ax.set_xlabel('Día del Mes')
        ax.set_ylabel('Ventas')
        ax.legend()
        st.pyplot(fig)
    
    
