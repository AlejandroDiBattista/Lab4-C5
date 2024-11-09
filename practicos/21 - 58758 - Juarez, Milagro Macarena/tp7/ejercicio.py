import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Parámetros de la interfaz
st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, 5)
entrenar = st.sidebar.button("Entrenar")

# Contenedor en el sidebar para mostrar el progreso de epochs
epoch_placeholder = st.sidebar.empty()  # Contenedor de texto para el número de epoch y error

## Crear Red Neuronal
class NeuralNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## Leer Datos
@st.cache_data
def load_data():
    data = pd.read_csv('ventas.csv')
    return data

data = load_data()

## Normalizar Datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values.reshape(-1, 1)
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

## Entrenar Red Neuronal
def train_model(model, X_tensor, y_tensor, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    progress_bar = st.progress(0)   # Barra de progreso visual
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Actualizar el contenedor en cada epoch
        if epoch % (epochs // 100) == 0 or epoch == epochs - 1:
            epoch_placeholder.text(f"Epoch {epoch + 1}/{epochs} - Error: {loss.item():.6f}")
            progress_bar.progress((epoch + 1) / epochs)
    
    progress_bar.progress(1.0)  # Asegurar que la barra llegue al 100% al finalizar
    return loss_history

if entrenar:
    # Inicializar modelo
    model = NeuralNet(hidden_neurons)
    
    # Entrenamiento
    st.write("Entrenando la red neuronal...")
    loss_history = train_model(model, X_tensor, y_tensor, learning_rate, epochs)
    st.sidebar.success("Entrenamiento exitoso")
    
    ## Guardar Modelo
    torch.save(model.state_dict(), "modelo_entrenado.pth")

    ## Graficar Predicciones
    with torch.no_grad():
        predicted = model(X_tensor).numpy()
        predicted = scaler_y.inverse_transform(predicted)
    
    # Gráfico de la evolución de la pérdida
    fig, ax = plt.subplots()
    ax.plot(loss_history, label='Pérdidas', color='green')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.legend()
    st.sidebar.pyplot(fig)
    
    # Gráfico de ventas y predicción
    plt.figure(figsize=(8, 6))
    plt.scatter(data['dia'], data['ventas'], color='blue', label="Datos Reales")
    plt.plot(data['dia'], predicted, color='red', label="Curva de Ajuste")
    plt.xlabel("Dia del Mes")
    plt.ylabel("Ventas")
    plt.legend()
    st.pyplot(plt)
