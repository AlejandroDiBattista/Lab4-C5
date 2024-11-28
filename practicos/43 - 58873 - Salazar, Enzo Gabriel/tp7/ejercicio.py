import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd  # Si estás trabajando con CSV, también necesitas pandas.
import streamlit as st  # Importación faltante

st.title("Estimación de Ventas Diarias")

class VentasNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # Corrección de __init__
        super(VentasNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)

# Panel de configuración de parámetros en la barra lateral
st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
num_epochs = st.sidebar.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=10)
hidden_units = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10)

# Funciones de normalización y desnormalización
def normalizar(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def desnormalizar(normalized_data, original_data):
    return normalized_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

# Cargar los datos de ventas y verificar su disponibilidad
@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv("ventas.csv")
    except FileNotFoundError:
        st.error("Error: No se encontró el archivo 'ventas.csv'.")
        return pd.DataFrame(columns=["dia", "ventas"])

# Preparación de datos
datos = cargar_datos()
if datos.empty:
    st.stop()

# Verificar las columnas del archivo de datos
if 'dia' not in datos.columns or 'ventas' not in datos.columns:
    st.error("Las columnas 'dia' y 'ventas' deben estar presentes en el archivo CSV.")
    st.stop()

# Configuración de los datos para el entrenamiento
dias = datos['dia'].values.reshape(-1, 1)
ventas = datos['ventas'].values.reshape(-1, 1)
dias_norm = normalizar(dias)
ventas_norm = normalizar(ventas)
x_tensor = torch.FloatTensor(dias_norm)
y_tensor = torch.FloatTensor(ventas_norm)

# Crear el modelo de red neuronal y configurar el optimizador y la función de pérdida
modelo = VentasNN(1, hidden_units, 1)
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

# Iniciar el entrenamiento al hacer clic en el botón
if st.sidebar.button("Entrenar"):
    barra_progreso = st.sidebar.progress(0)
    historial_perdida = []
    for epoca in range(int(num_epochs)):
        modelo.train()
        optimizador.zero_grad()
        # Hacer predicciones y calcular la pérdida
        predicciones = modelo(x_tensor)
        perdida = criterio(predicciones, y_tensor)
        # Optimización de parámetros
        perdida.backward()
        optimizador.step()
        # Guardar el historial de pérdida para graficar después
        historial_perdida.append(perdida.item())
        barra_progreso.progress((epoca + 1) / num_epochs)

    # Mensaje de finalización y visualización de la última pérdida
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Epoch {num_epochs}/{num_epochs} — Error: {perdida.item():.6f}")

    # Graficar la evolución de la pérdida durante el entrenamiento
    plt.figure(figsize=(5, 3))
    plt.plot(historial_perdida, color='green', label='Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Historial de Pérdida')
    plt.legend()
    st.sidebar.pyplot(plt)

    # Realizar predicciones finales y desnormalizarlas para compararlas con los datos reales
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(x_tensor).numpy()
        predicciones_desnormalizadas = desnormalizar(predicciones, ventas)

    # Gráfico de las ventas reales y predicciones de la red neuronal
    plt.figure(figsize=(8, 6))
    plt.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos Reales')
    plt.plot(datos['dia'], predicciones_desnormalizadas, color='red', label='Curva de Ajuste')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.title('Estimación de Ventas Diarias')
    plt.legend()
    st.pyplot(plt)
    
