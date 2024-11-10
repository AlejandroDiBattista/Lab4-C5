import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
st.title("Estimación de Ventas Diarias")
st.sidebar.header("Parámetros de Entrenamiento")

# Parámetros de entrada
learning_rate = st.sidebar.number_input("Aprendizaje", value=0.01, step=0.001, format="%.4f")
epochs = st.sidebar.number_input("Repeticiones", value=1000, step=100)
hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", value=10, step=1)

# Entrenamiento al hacer clic en el botón
if st.sidebar.button("Entrenar"):
    # Preparación de datos
    X = torch.tensor(df[['dia']].values, dtype=torch.float32)
    y = torch.tensor(df[['ventas_scaled']].values, dtype=torch.float32)
    
    # Inicialización del modelo y optimizador
    model = SalesPredictor(input_dim=1, hidden_dim=hidden_neurons, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Entrenamiento del modelo
    loss_history = []
    progress = st.sidebar.progress(0)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress.progress((epoch + 1) / epochs)
        
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {loss.item():.6f}")

    # Gráfico de pérdida
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_history, color="green")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    ax_loss.set_title("Pérdida")
    st.sidebar.pyplot(fig_loss)
    
    # Predicción y desescalado
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X).numpy()
    pred_inversed = scaler.inverse_transform(pred_scaled)

    # Gráfico de ventas reales vs predicción
    fig, ax = plt.subplots()
    ax.plot(df['dia'], df['ventas'], 'o', color="blue", label="Datos Reales")
    ax.plot(df['dia'], pred_inversed, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
