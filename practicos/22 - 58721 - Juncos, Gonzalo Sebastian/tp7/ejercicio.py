import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuración de estilo para Streamlit
st.set_page_config(layout="wide")

# Cargar y escalar los datos
df = pd.read_csv('ventas.csv')
scaler = MinMaxScaler()
df['ventas_scaled'] = scaler.fit_transform(df[['ventas']])

# Definición de la red neuronal
class SalesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SalesPredictor, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Configuración de la interfaz de Streamlit
st.sidebar.title("Parámetros de Entrenamiento")
learning_rate = st.sidebar.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, step=0.01, value=0.1, format="%.2f")
epochs = st.sidebar.number_input("Cantidad de Épocas", min_value=10, max_value=10000, step=10, value=100)
hidden_neurons = st.sidebar.number_input("Neurona Capa Oculta", min_value=1, max_value=100, step=1, value=5)

# Entrenamiento al hacer clic en el botón
if st.sidebar.button("Entrenar"):
    # Preparación de datos
    X = torch.tensor(df[['dia']].values, dtype=torch.float32)
    y = torch.tensor(df[['ventas_scaled']].values, dtype=torch.float32)
    
    # Inicialización del modelo y optimizador
    model = SalesPredictor(input_dim=1, hidden_dim=hidden_neurons, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Cambié a Adam para mejor convergencia
    
    # Entrenamiento del modelo
    loss_history = []
    progress = st.progress(0)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % (epochs // 10) == 0:  # Actualiza la barra de progreso cada 10%
            progress.progress((epoch + 1) / epochs)
        
    st.success("Entrenamiento exitoso")
    st.write(f"Epoch {epochs}/{epochs} - Error: {loss.item():.5f}")

    # Gráfico de pérdida (pérdida vs épocas) en la barra lateral
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_history, color="green", label="Pérdida")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.set_title("Pérdida durante el Entrenamiento")
    st.sidebar.pyplot(fig_loss)

    # Predicción y desescalado
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X).numpy()
    pred_inversed = scaler.inverse_transform(pred_scaled)

    # Gráfico de ventas reales vs predicción (en la sección principal)
    fig_sales, ax_sales = plt.subplots()
    ax_sales.plot(df['dia'], df['ventas'], 'o', color="blue", label="Datos Reales", markersize=5)
    ax_sales.plot(df['dia'], pred_inversed, color="red", label="Curva de Ajuste", linewidth=2)
    ax_sales.set_xlabel("Día del Mes")
    ax_sales.set_ylabel("Ventas")
    ax_sales.set_title("Estimación de Ventas Diarias", fontsize=16, fontweight='bold')
    ax_sales.legend()
    st.pyplot(fig_sales)

