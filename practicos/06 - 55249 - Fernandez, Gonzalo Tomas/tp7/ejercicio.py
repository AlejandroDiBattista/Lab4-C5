import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

st.title("Estimación de Ventas Diarias")

## Crear Red Neuronal
class VentasNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

# Agrupar "Aprendizaje" y "Repeticiones" en la misma fila
st.sidebar.header("Parámetros de Entrenamiento")
col1, col2 = st.sidebar.columns(2)

# Configuración con botones de incremento y decremento
learning_rate = col1.number_input("Aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.3f")
epochs = col2.number_input("Repeticiones", min_value=10, max_value=10000, value=100, step=10)

# "Neuronas Capa Oculta" en una línea independiente
hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=5, step=1)

## Función para normalizar datos usando solo numpy
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

## Desnormalizar predicciones
def denormalize(norm_data, original_data):
    return norm_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

## Leer Datos
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("ventas.csv")
        return data
    except FileNotFoundError:
        st.error("Archivo 'ventas.csv' no encontrado.")
        return pd.DataFrame(columns=["dia", "ventas"])

## Cargar los datos
data = load_data()
days = data['dia'].values.reshape(-1, 1)
sales = data['ventas'].values.reshape(-1, 1)

## Normalizar los datos
days_norm = normalize(days)
sales_norm = normalize(sales)

## Convertir a tensores de PyTorch
x_train = torch.FloatTensor(days_norm)
y_train = torch.FloatTensor(sales_norm)

## Instanciar la red neuronal
model = VentasNN(1, hidden_neurons, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## Entrenar Red Neuronal
if st.sidebar.button("Entrenar"):
    progress_bar = st.sidebar.progress(0)
    loss_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        ## Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        ## Backward pass y optimización
        loss.backward()
        optimizer.step()

        ## Guardar Modelo
        loss_list.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
    # Actualizar barra de progreso con el número de épocas y error actual
    st.sidebar.text(f"Epoch {epoch + 1}/{epochs} - Error: {loss.item():.6f}")
    st.sidebar.success("Entrenamiento exitoso")

    ## Gráfico de evolución de pérdida
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(loss_list, color="green", label="Pérdidas")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.sidebar.pyplot(fig)

    ## Generar predicciones
    model.eval()
    with torch.no_grad():
        predictions = model(x_train).numpy()
        predictions_denorm = denormalize(predictions, sales)

    ## Graficar Predicciones
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(data['dia'], data['ventas'], color="blue", label="Datos Reales")
    ax.plot(data['dia'], predictions_denorm, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
