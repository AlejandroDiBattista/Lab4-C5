import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Crear Red Neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

## Leer Datos

def leer_datos():
    data = pd.read_csv('ventas.csv')
    return data
## Normalizar Datos

def normalizar_datos(data):
    x = data["dia"].values.reshape(-1, 1)
    y = data["ventas"].values.reshape(-1, 1)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return x, y, x_norm, y_norm, x_min, x_max, y_min, y_max

## Entrenar Red Neuronal

def entrenar_red(modelo, x_tensor, y_tensor, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        y_pred = modelo(x_tensor)
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        barra_progreso.progress((epoch + 1) / epochs)
    
    progreso = f"Epoch {epoch + 1}/{epochs}, Error: {loss.item():.4f}"
    st.sidebar.text(progreso)

    return loss_history

## Graficar Predicciones
def graficar_resultados(x, y, x_pred, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Datos reales', color='blue')
    plt.plot(x_pred, y_pred, label='Predicciones', color='red')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

st.title('Estimación de Ventas Diarias')
st.sidebar.header('Parámetros de Entrenamiento')

learning_rate = st.sidebar.slider('Tasa de Aprendizaje', 0.0001, 0.1, 0.01)
epochs = st.sidebar.slider('Número de Épocas', 10, 1000, 100)
hidden_size = st.sidebar.slider('Neuronas en Capa Oculta', 1, 100, 10)

if st.sidebar.button('Entrenar'):
    data = leer_datos()
    x, y, x_norm, y_norm, x_min, x_max, y_min, y_max = normalizar_datos(data)
    
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    
    modelo = NeuralNetwork(1, hidden_size, 1)
    barra_progreso = st.sidebar.progress(0)
    loss_history = entrenar_red(modelo, x_tensor, y_tensor, epochs, learning_rate)
    
    st.sidebar.success('Entrenamiento Completado')

    plt.figure()
    plt.plot(loss_history, label='Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    st.sidebar.pyplot(plt)

    x_pred = np.linspace(1, 31, 100).reshape(-1, 1)
    x_pred_norm = (x_pred - x_min) / (x_max - x_min)
    x_pred_tensor = torch.tensor(x_pred_norm, dtype=torch.float32)
    y_pred = modelo(x_pred_tensor).detach().numpy()
    y_pred_rescaled = y_pred * (y_max - y_min) + y_min

    graficar_resultados(x, y, x_pred, y_pred_rescaled)