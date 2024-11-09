import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Clase para la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, entradas, ocultas):
        super(RedNeuronal, self).__init__()
        self.linear1 = nn.Linear(entradas, ocultas)
        self.linear2 = nn.Linear(ocultas, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

# Cargar datos y normalizar entre 0 y 1
datos = pd.read_csv('ventas.csv')
x = datos['dia'].values.reshape(-1, 1)
y = datos['ventas'].values.reshape(-1, 1)
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

x_normalizado = (x_tensor - x_tensor.min()) / (x_tensor.max() - x_tensor.min())
y_normalizado = (y_tensor - y_tensor.min()) / (y_tensor.max() - y_tensor.min())

# Interfaz de Streamlit
st.title('Estimación de Ventas Diarias')
st.sidebar.header('Parámetros de la Red Neuronal')
tasa_aprendizaje = st.sidebar.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1)
epocas = st.sidebar.slider('Cantidad de Épocas', 10, 10000, 100)
neuronas_ocultas = st.sidebar.slider('Neurones en la Capa Oculta', 1, 100, 5)

st.subheader('Datos de Ventas Diarias')
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Ventas reales')
plt.xlabel('Día')
plt.ylabel('Ventas')
st.pyplot(plt)

# Entrenamiento
if st.sidebar.button('Entrenar'):
    modelo = RedNeuronal(1, neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)

    progreso = st.progress(0)
    perdida_por_epoca = []

    for epoca in range(epocas):
        prediccion = modelo(x_normalizado)
        perdida = criterio(prediccion, y_normalizado)
        
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        perdida_por_epoca.append(perdida.item())
        
        if (epoca + 1) % (epocas // 100) == 0:  # Para mostrar progreso cada 1%
            progreso.progress((epoca + 1) / epocas)

    st.success('¡Entrenamiento finalizado!')

    st.subheader('Evolución de la Función Pérdida')
    plt.figure(figsize=(10, 5))
    plt.plot(range(epocas), perdida_por_epoca, label='Función de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Costo')
    st.pyplot(plt)

    st.subheader('Estimación de Ventas Diarias con Red Neuronal')
    prediccion_final = modelo(x_normalizado).detach().numpy() * (y_tensor.max().item() - y_tensor.min().item()) + y_tensor.min().item()
    plt.figure(figsize=(10, 5))
    plt.scatter(x.flatten(), y.flatten(), label='Ventas reales')
    plt.plot(x, prediccion_final, color='red', label='Predicción')
    plt.xlabel('Día')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)
