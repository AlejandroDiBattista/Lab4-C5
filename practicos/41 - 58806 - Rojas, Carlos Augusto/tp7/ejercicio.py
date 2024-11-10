import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Leer Datos
data = pd.read_csv('ventas.csv')
X = data['dia'].values.reshape(-1, 1)
y = data['ventas'].values.reshape(-1, 1)

## Normalizar Datos
X_min, X_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
X_normalized = (X - X_min) / (X_max - X_min)
y_normalized = (y - y_min) / (y_max - y_min)

## Crear Red Neuronal
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

st.sidebar.header('Parámetros de Entrenamiento')
learning_rate = st.sidebar.number_input('Aprendizaje', min_value=0.0001, max_value=1.0, value=0.1, step=0.1)
epochs = st.sidebar.number_input('Repeticiones', min_value=10, max_value=10000, value=100)
hidden_neurons = st.sidebar.number_input('Neuronas Capa Oculta', min_value=1, max_value=100, value=5)

## Entrenar Red Neuronal
if st.sidebar.button('Entrenar'):
    # Crear el modelo
    model = NeuralNet(input_size=1, hidden_size=hidden_neurons, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    X_tensor = torch.Tensor(X_normalized)
    y_tensor = torch.Tensor(y_normalized)
    losses = []

    # Barra de progreso
    progress_bar = st.sidebar.progress(0)
    for epoch in range(epochs):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Actualizar barra de progreso
        progress_bar.progress((epoch + 1) / epochs)
    
    st.sidebar.success('Entrenamiento exitoso')
    st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {loss.item():.6f}")
    plt.style.use("default")

    ## Graficar Pérdida durante el Entrenamiento
    st.sidebar.subheader("Perdida durante el entrenamiento")
    plt.figure()
    plt.plot(range(epochs), losses, 'g-', label="Perdidas")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida")
    plt.legend()
    st.sidebar.pyplot(plt)  # Colocamos la gráfica de pérdida en la sidebar

    # Guardar modelo
    torch.save(model.state_dict(), 'modelo_entrenado.pth')

    ## Graficar Predicciones vs. Datos Reales
    model.eval()  # Cambia el modelo a modo de evaluación
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Desnormalizar las predicciones
    predictions = predictions * (y_max - y_min) + y_min
    
    st.subheader("Estimación de Ventas Diarias")
    plt.figure()
    plt.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
    plt.plot(data['dia'], predictions, color='red', label='Curva de Ajuste')
    plt.xlabel("Dia del Mes")
    plt.ylabel("Ventas")
    plt.legend()
    st.pyplot(plt)
