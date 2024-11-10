#se debe cargar el archivo csv
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias con Redes Neuronales')

uploaded_file = st.file_uploader("Sube el archivo de ventas (.csv)", type="csv")

st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en capa oculta", 1, 100, 5)

class VentasNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def preprocess_data(data):
    scaler = MinMaxScaler()
    data[['día']] = scaler.fit_transform(data[['día']])
    data[['ventas']] = scaler.fit_transform(data[['ventas']])
    return data, scaler

def train_model(model, data, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    for epoch in range(epochs):
        inputs = torch.tensor(data[['día']].values, dtype=torch.float32)
        labels = torch.tensor(data[['ventas']].values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        st.sidebar.progress((epoch + 1) / epochs)
    return loss_history

def plot_results(data, model, scaler):
    inputs = torch.tensor(data[['día']].values, dtype=torch.float32)
    predictions = model(inputs).detach().numpy()

    data['Predicción'] = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(data['día'] * 30, scaler.inverse_transform(data[['ventas']]), label="Ventas Reales", color="blue")
    plt.plot(data['día'] * 30, data['Predicción'], label="Predicción", color="red")
    plt.xlabel("Día")
    plt.ylabel("Ventas")
    plt.legend()
    st.pyplot(plt)

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, delimiter=",")
        data.columns = data.columns.str.strip()  
        data.rename(columns={"dia": "día", "ventas": "ventas"}, inplace=True)

        st.write("Columnas después de renombrar:", data.columns)
        st.write("Muestra de los datos cargados:", data.head())

        if 'día' not in data.columns or 'ventas' not in data.columns:
            st.error("El archivo CSV debe contener las columnas 'día' y 'ventas'")
        else:
            data, scaler = preprocess_data(data)

            model = VentasNN(input_size=1, hidden_size=hidden_neurons, output_size=1)
            loss_history = train_model(model, data, epochs, learning_rate)

            st.success("Entrenamiento completado con éxito.")

            plt.figure(figsize=(10, 5))
            plt.plot(range(epochs), loss_history, label="Función de costo (MSE)")
            plt.xlabel("Épocas")
            plt.ylabel("MSE")
            plt.legend()
            st.pyplot(plt)

            plot_results(data, model, scaler)
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV para continuar.")
