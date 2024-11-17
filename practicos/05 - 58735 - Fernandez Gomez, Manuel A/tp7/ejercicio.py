import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



st.title('Estimación de Ventas Diarias')


st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")


def load_data():
    data = pd.read_csv('ventas.csv')
    return data

data = load_data()
st.write("Datos de Ventas Diarias")
st.line_chart(data['ventas'])


scaler = MinMaxScaler()
data['ventas'] = scaler.fit_transform(data[['ventas']])
x = torch.tensor(data['día'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)


class SalesPredictionModel(nn.Module):
    def __init__(self, hidden_neurons):
        super(SalesPredictionModel, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


def train_model(model, x, y, learning_rate, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_history = []
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
    return loss_history


if train_button:
    model = SalesPredictionModel(hidden_neurons)
    loss_history = train_model(model, x, y, learning_rate, epochs)
    
    
    st.success("Entrenamiento finalizado con éxito.")
    
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Función de Costo")
    ax.set_title("Evolución de la Función de Costo")
    st.pyplot(fig)
    
    
    model.eval()
    predictions = model(x).detach().numpy()
    
    
    fig, ax = plt.subplots()
    ax.plot(data['día'], scaler.inverse_transform(y.numpy()), label="Datos reales")
    ax.plot(data['día'], scaler.inverse_transform(predictions), label="Predicción", linestyle="--")
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.set_title("Ventas Reales vs Predicción")
    ax.legend()
    st.pyplot(fig)
