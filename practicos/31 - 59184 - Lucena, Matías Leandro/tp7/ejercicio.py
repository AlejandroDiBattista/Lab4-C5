import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
data = pd.read_csv('ventas.csv')
dias = data['dia'].values.reshape(-1, 1)
ventas = data['ventas'].values.reshape(-1, 1)

# Normalizar datos
scaler = MinMaxScaler()
dias_norm = scaler.fit_transform(dias)
ventas_norm = scaler.fit_transform(ventas)

# Definir red neuronal
class VentasNN(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

# Configurar interfaz
st.title("Estimación de Ventas Diarias")
st.sidebar.header("Parámetros de Entrenamiento")

# Parámetros de entrada
learning_rate = float(st.sidebar.text_input("Tasa de Aprendizaje", "0.1"))
epochs = st.sidebar.number_input("Cantidad de Épocas", 10, 10000, 100, 10)
hidden_neurons = st.sidebar.number_input("Neurona Capa Oculta", 1, 100, 5, 1)

# Entrenamiento
if st.sidebar.button("Entrenar"):
    dias_tensor = torch.FloatTensor(dias_norm)
    ventas_tensor = torch.FloatTensor(ventas_norm)

    model = VentasNN(hidden_neurons)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    loss_values = []
    progress_bar = st.sidebar.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(dias_tensor)
        loss = criterion(output, ventas_tensor)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
    
    st.sidebar.success("Entrenamiento exitoso")

    # Graficar pérdida
    fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
    ax_loss.plot(loss_values, color="green", label="Pérdidas")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.legend()
    st.sidebar.pyplot(fig_loss)

    # Predicción
    with torch.no_grad():
        ventas_pred = model(dias_tensor).numpy()
    ventas_pred = scaler.inverse_transform(ventas_pred)
    dias = dias.flatten()
    ventas = ventas.flatten()

    # Graficar resultados
    fig_sales, ax_sales = plt.subplots(figsize=(8, 6))
    ax_sales.scatter(dias, ventas, color="blue", label="Datos Reales")
    ax_sales.plot(dias, ventas_pred, color="red", label="Curva de Ajuste")
    ax_sales.set_xlabel("Día del Mes")
    ax_sales.set_ylabel("Ventas")
    ax_sales.legend()
    st.pyplot(fig_sales)
