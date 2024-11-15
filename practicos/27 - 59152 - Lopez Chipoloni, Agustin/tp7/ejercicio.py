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

# Cargar dato

data = pd.read_csv('ventas.csv')

# Visualización inicial de datos
st.title("Exploración de Clasificación con Redes Neuronales")
st.write("Datos de ventas diarias")
st.line_chart(data['ventas'])

# Panel de entrada
st.sidebar.title("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en capa oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

# Clase de red neuronal
class SalesNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(SalesNet, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Función de entrenamiento sin torch.optim
def train_model(model, data, epochs, learning_rate):
    X = torch.tensor(data[['dia']].values, dtype=torch.float32)
    y = torch.tensor(data[['ventas']].values, dtype=torch.float32)
    criterion = nn.MSELoss()
    cost_history = []

    progress_bar = st.sidebar.progress(0)
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        output = model(X)
        loss = criterion(output, y)
        
        # Backward pass y actualización manual de pesos
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad  # Actualización manual
            
        model.zero_grad()  # Limpia los gradientes
        
        cost_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    progress_bar.empty()
    return cost_history

# Entrenamiento y visualización
if train_button:
    model = SalesNet(hidden_neurons)
    cost_history = train_model(model, data, epochs, learning_rate)
    
    st.success("Entrenamiento completado!")
    
    # Visualizar predicción
    with torch.no_grad():
        X = torch.tensor(data[['dia']].values, dtype=torch.float32)
        prediction = model(X).numpy()
    
    fig, ax = plt.subplots()
    ax.plot(data['dia'], data['ventas'], label="Datos Reales")
    ax.plot(data['dia'], prediction, label="Predicción", linestyle="--")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    # Visualizar evolución de la función de costo
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Épocas")
    plt.ylabel("Costo")
    plt.title("Evolución del Costo")
    st.pyplot(plt)