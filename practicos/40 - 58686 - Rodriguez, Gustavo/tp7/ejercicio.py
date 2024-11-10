import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.title("Predicción de Ventas Diarias")

class SalesNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SalesNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)

st.sidebar.header("Configuración de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1, 0.01)
num_epochs = st.sidebar.slider("Número de Épocas", 10, 10000, 100, 10)
hidden_units = st.sidebar.slider("Unidades en Capa Oculta", 1, 100, 5)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize(normalized_data, original_data):
    return normalized_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("ventas.csv")
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo 'ventas.csv'.")
        return pd.DataFrame(columns=["dia", "ventas"])


data = load_data()
if data.empty:
    st.stop()


if 'dia' not in data.columns or 'ventas' not in data.columns:
    st.error("Las columnas 'dia' y 'ventas' deben estar presentes en el archivo CSV.")
    st.stop()


days = data['dia'].values.reshape(-1, 1)
sales = data['ventas'].values.reshape(-1, 1)
days_normalized = normalize(days)
sales_normalized = normalize(sales)


x_train_tensor = torch.FloatTensor(days_normalized)
y_train_tensor = torch.FloatTensor(sales_normalized)


model = SalesNN(1, hidden_units, 1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if st.sidebar.button("Entrenar Modelo"):
    progress_bar = st.sidebar.progress(0)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

   
        predictions = model(x_train_tensor)
        loss = loss_function(predictions, y_train_tensor)

    
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / num_epochs)

    st.sidebar.success("Entrenamiento completado.")
    st.sidebar.text(f"Última Pérdida: {loss.item():.6f}")

    plt.figure(figsize=(5, 3))
    plt.plot(loss_history, color='orange', label='Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la Pérdida durante el Entrenamiento')
    plt.legend()
    st.sidebar.pyplot(plt)

    model.eval()
    with torch.no_grad():
        predictions = model(x_train_tensor).numpy()
        predictions_denormalized = denormalize(predictions, sales)

    plt.figure(figsize=(7, 5))
    plt.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
    plt.plot(data['dia'], predictions_denormalized, color='red', label='Predicciones')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.title('Predicción de Ventas Diarias')
    plt.legend()
    st.pyplot(plt)
