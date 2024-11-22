import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


st.set_page_config(page_title="Estimación de Ventas Diarias", layout="wide")


st.title("Estimación de Ventas Diarias")


@st.cache_data
def load_data():
    try:
        
        data = pd.read_csv("ventas.csv")
        
        data.columns = data.columns.str.strip()
       
        data.rename(columns={"dia": "día", "sales": "ventas"}, inplace=True)
        return data
    except FileNotFoundError:
        st.error("El archivo 'ventas.csv' no se encontró. Por favor, asegúrate de colocarlo en el directorio del script.")
        return None


def train_neural_network(data, learning_rate, epochs, hidden_neurons):
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    dias_normalizados = scaler_x.fit_transform(data[['día']])
    ventas_normalizadas = scaler_y.fit_transform(data[['ventas']])

    x_train = torch.FloatTensor(dias_normalizados)
    y_train = torch.FloatTensor(ventas_normalizadas)

    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.hidden = nn.Linear(1, hidden_neurons)
            self.output = nn.Linear(hidden_neurons, 1)
            self.activation = nn.ReLU()

        def forward(self, x):
            x = self.activation(self.hidden(x))
            x = self.output(x)
            return x

    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    
    loss_history = []
    progress_bar = st.progress(0)

    for epoch in tqdm(range(epochs)):
        
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    progress_bar.empty()
    return model, loss_history, scaler_x, scaler_y

st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100, 10)
hidden_neurons = st.sidebar.slider("Neuronas en la capa oculta", 1, 100, 5, 1)
train_button = st.sidebar.button("Entrenar")


data = load_data()

if data is not None:
   
    st.subheader("Datos de ventas")
    st.write(data)

   
    fig, ax = plt.subplots()
    ax.scatter(data["día"], data["ventas"], color="blue", label="Datos Reales")
    ax.set_title("Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    if train_button:
        
        st.subheader("Entrenamiento en curso...")
        model, loss_history, scaler_x, scaler_y = train_neural_network(data, learning_rate, epochs, hidden_neurons)

        st.success("Entrenamiento exitoso")

      
        fig, ax = plt.subplots()
        ax.plot(loss_history, color="green", label="Pérdidas")
        ax.set_title("Evolución de la Pérdida")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Pérdida")
        ax.legend()
        st.pyplot(fig)

       
        dias = torch.FloatTensor(scaler_x.transform(data[["día"]]))
        predicciones = model(dias).detach().numpy()
        predicciones = scaler_y.inverse_transform(predicciones)

        
        fig, ax = plt.subplots()
        ax.scatter(data["día"], data["ventas"], color="blue", label="Datos Reales")
        ax.plot(data["día"], predicciones, color="red", label="Curva de Ajuste")
        ax.set_title("Estimación de Ventas Diarias")
        ax.set_xlabel("Día del Mes")
        ax.set_ylabel("Ventas")
        ax.legend()
        st.pyplot(fig)
