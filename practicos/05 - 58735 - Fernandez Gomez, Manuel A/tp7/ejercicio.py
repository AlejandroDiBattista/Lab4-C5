import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Cargar el archivo de datos y realizar el escalado
data = pd.read_csv('ventas.csv')
scaler = StandardScaler()
data['ventas_normalizadas'] = scaler.fit_transform(data[['ventas']])

## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias')
# Redefinir la clase de la red neuronal con un nuevo diseño
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):  # Cambiado _init_ por __init__
        super(NeuralNet, self).__init__()  # Cambiado _init_ por __init__
        self.input_layer = nn.Linear(input_size, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_size)

    def forward(self, inputs):
        activation = torch.relu(self.input_layer(inputs))
        return self.output_layer(activation)

# Configuración del panel interactivo de Streamlit
st.header("Predicción de Ventas Diarias")
st.sidebar.title("Configuración del Modelo")

# Configuración interactiva de los parámetros
learning_rate = st.sidebar.number_input("Tasa de Aprendizaje", value=0.01, step=0.001, format="%.4f")
iterations = st.sidebar.number_input("Número de Iteraciones", value=1000, step=100)
hidden_units = st.sidebar.number_input("Unidades en Capa Oculta", value=12, step=1)

# Lógica de entrenamiento del modelo al hacer clic en el botón
if st.sidebar.button("Iniciar Entrenamiento"):
    # Preparación de los datos
    features = torch.tensor(data[['dia']].values, dtype=torch.float32)
    target = torch.tensor(data[['ventas_normalizadas']].values, dtype=torch.float32)

    # Inicialización del modelo y el optimizador
    model = NeuralNet(input_size=1, hidden_units=hidden_units, output_size=1)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Proceso de entrenamiento
    loss_values = []
    progress_bar = st.sidebar.progress(0)

    for epoch in range(iterations):
        model.train()
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, target)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        progress_bar.progress((epoch + 1) / iterations)

    st.sidebar.success("Entrenamiento Completo")
    st.sidebar.write(f"Iteración {iterations}/{iterations} - Error: {loss.item():.6f}")

    # Mostrar gráfico de la función de pérdida
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_values, color="red")
    ax_loss.set_xlabel("Iteraciones")
    ax_loss.set_ylabel("Error Cuadrático Medio")
    ax_loss.set_title("Evolución de la Pérdida")
    st.sidebar.pyplot(fig_loss)

    # Realizar predicciones y revertir el escalado
    model.eval()
    with torch.no_grad():
        scaled_predictions = model(features).numpy()
    original_predictions = scaler.inverse_transform(scaled_predictions)

    # Gráfico de comparación entre datos reales y predicciones
    fig, ax = plt.subplots()
    ax.plot(data['dia'], data['ventas'], 'o', color="blue", label="Ventas Reales")
    ax.plot(data['dia'], original_predictions, color="orange", label="Ventas Predichas")
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
