import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

## Verificar si el archivo existe
file_path = "ventas.csv"  # Cambia a la ruta completa si es necesario

# Verificación de existencia del archivo
if os.path.exists(file_path):
    st.write("El archivo 'ventas.csv' fue encontrado.")
else:
    st.error("El archivo 'ventas.csv' NO fue encontrado. Verifica la ruta y asegúrate de que el archivo esté en el directorio correcto.")
    st.stop()  # Detener la ejecución si el archivo no existe

## Crear Red Neuronal
class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

## Leer Datos
def cargar_datos():
    datos = pd.read_csv(file_path)
    return datos

datos = cargar_datos()

x_datos = datos[['dia']].values
y_datos = datos[['ventas']].values

## Normalizar y Convertir Datos a Tensores
min_x, max_x = x_datos.min(), x_datos.max()
min_y, max_y = y_datos.min(), y_datos.max()

# Normalización
X_normalized = (x_datos - min_x) / (max_x - min_x)
y_normalized = (y_datos - min_y) / (max_y - min_y)

# Conversión a tensores de PyTorch con la forma correcta
X_tensor = torch.tensor(X_normalized, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32).reshape(-1, 1)

## Entrenar Red Neuronal
def train_model(learning_rate, epochs, hidden_neurons):
    model = SimpleNN(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    progress_bar = st.progress(0)
    loss_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    return model, loss_values

## Guardar Modelo
def save_model(model, filename="modelo_ventas.pth"):
    torch.save(model.state_dict(), filename)
    st.write(f"Modelo guardado como {filename}")

## Cargar Modelo (opcional para pruebas)
def load_model(filename="modelo_ventas.pth", hidden_neurons=5):
    model = SimpleNN(hidden_neurons)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

## Graficar Predicciones
def plot_results(X, y, y_pred, loss_values):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Gráfico de pérdida
    ax1.plot(loss_values, label="Pérdida")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.legend()

    # Gráfico de predicción
    ax2.scatter(X, y, color='blue', label="Datos Reales")
    ax2.plot(X, y_pred, color='red', label="Curva de Ajuste")
    ax2.set_xlabel("Día del Mes")
    ax2.set_ylabel("Ventas")
    ax2.legend()

    st.pyplot(fig)

## Interfaz de Streamlit
st.title('Estimación de Ventas Diarias')

# Panel de control para ingresar los parámetros de la red neuronal
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en la capa oculta", 1, 100, 5)

if st.sidebar.button("Entrenar"):
    # Entrenar el modelo con los parámetros ingresados
    model, loss_values = train_model(learning_rate, epochs, hidden_neurons)

    # Generar predicciones
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()

    # Mostrar el mensaje de éxito y los gráficos
    st.success("Entrenamiento exitoso")
    plot_results(x_datos, y_datos, y_pred, loss_values)

