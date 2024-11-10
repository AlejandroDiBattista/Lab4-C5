import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Definición de la arquitectura de la red neuronal
class ModeloVentas(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModeloVentas, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        output = self.fc2(x)
        return output

# Cargar y normalizacion de datos
data = pd.read_csv('ventas.csv')
dias = data['dia'].values.reshape(-1, 1)
ventas = data['ventas'].values.reshape(-1, 1)

dias_tensor = torch.tensor(dias, dtype=torch.float32)
ventas_tensor = torch.tensor(ventas, dtype=torch.float32)

dias_norm = (dias_tensor - dias_tensor.min()) / (dias_tensor.max() - dias_tensor.min())
ventas_norm = (ventas_tensor - ventas_tensor.min()) / (ventas_tensor.max() - ventas_tensor.min())

# Interfaz de streamLit
st.title('Predicción de Ventas Diarias')
st.sidebar.header('Parámetros del Modelo')
learning_rate = st.sidebar.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1)
num_epochs = st.sidebar.slider('Número de Épocas', 10, 10000, 100)
num_neuronas = st.sidebar.slider('Neurones en la Capa Oculta', 1, 100, 5)

st.subheader('Datos de Ventas Diarias')
plt.figure(figsize=(10, 5))
plt.scatter(dias, ventas, label='Ventas observadas')
plt.xlabel('Día')
plt.ylabel('Ventas')
st.pyplot(plt)

# Entrenamiento
if st.sidebar.button('Entrenar Modelo'):
    red_neuronal = ModeloVentas(1, num_neuronas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(red_neuronal.parameters(), lr=learning_rate)

    progreso_entrenamiento = st.progress(0)
    historial_perdida = []

    for epoch in range(num_epochs):
        predicciones = red_neuronal(dias_norm)
        loss = criterio(predicciones, ventas_norm)
        
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()
        
        historial_perdida.append(loss.item())
        
        if (epoch + 1) % (num_epochs // 100) == 0:
            progreso_entrenamiento.progress((epoch + 1) / num_epochs)

    st.success('¡Entrenamiento completado!')

    st.subheader('Evolución de la Pérdida durante el Entrenamiento')
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), historial_perdida, label='Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    st.pyplot(plt)

    st.subheader('Predicción de Ventas con el Modelo Entrenado')
    ventas_predichas = red_neuronal(dias_norm).detach().numpy() * (ventas_tensor.max().item() - ventas_tensor.min().item()) + ventas_tensor.min().item()
    plt.figure(figsize=(10, 5))
    plt.scatter(dias.flatten(), ventas.flatten(), label='Ventas observadas')
    plt.plot(dias, ventas_predichas, color='red', label='Predicción del Modelo')
    plt.xlabel('Día')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)
