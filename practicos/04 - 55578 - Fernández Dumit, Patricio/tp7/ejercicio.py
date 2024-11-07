import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

datos = pd.read_csv("ventas.csv")
dias = datos['dia'].values.reshape(-1, 1)
ventas = datos['ventas'].values.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
dias = scaler_x.fit_transform(dias)
ventas = scaler_y.fit_transform(ventas)

class RedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

st.title("Estimación de Ventas Diarias")

st.sidebar.header("Parámetros de Entrenamiento")

col1, col2 = st.sidebar.columns(2)  

with col1:
    learning_rate = st.number_input("Aprendizaje", min_value=0.0, max_value=1.0, value=0.0100, step=0.01)

with col2:
    epochs = st.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=10)

hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10, step=1)

train_button = st.sidebar.button("Entrenar")

if train_button:
    input_size = 1
    output_size = 1
    model = RedNeuronal(input_size, hidden_neurons, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    dias_tensor = torch.tensor(dias, dtype=torch.float32)
    ventas_tensor = torch.tensor(ventas, dtype=torch.float32)
    
    loss_history = []
    progress_bar = st.sidebar.progress(0)
    final_error_text = st.sidebar.empty()  

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(dias_tensor)
        loss = criterion(outputs, ventas_tensor)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        progress_bar.progress((epoch + 1) / epochs)
    
    final_error_text.text(f"Epoch {epochs}/{epochs} - Error: {loss_history[-1]:.6f}")
    
    st.sidebar.success("Entrenamiento exitoso")
    
    fig, ax = plt.subplots()
    ax.plot(loss_history, color="green", label="Pérdidas")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.sidebar.pyplot(fig)

    model.eval()
    with torch.no_grad():
        predicciones = model(dias_tensor).numpy()
    predicciones = scaler_y.inverse_transform(predicciones)
    dias_original = scaler_x.inverse_transform(dias)
    ventas_original = scaler_y.inverse_transform(ventas)

    fig, ax = plt.subplots()
    ax.scatter(dias_original, ventas_original, color="blue", label="Datos Reales")
    ax.plot(dias_original, predicciones, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

