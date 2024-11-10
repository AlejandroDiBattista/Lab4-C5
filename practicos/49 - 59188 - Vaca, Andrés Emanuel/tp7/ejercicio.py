import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title('Exploración de Parámetros en Red Neuronal para Ventas Diarias')


col1, col2 = st.columns([1, 2])


with col1:
    st.header("Parámetros de Entrenamiento")
    learning_rate = st.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1, 0.01)
    epochs = st.slider("Cantidad de épocas", 10, 10000, 100, 100)
    hidden_neurons = st.slider("Neuronas en la capa oculta", 1, 100, 5)
    train_button = st.button("Entrenar")


try:
    data = pd.read_csv('ventas.csv')
except FileNotFoundError:
    st.error("No se pudo encontrar el archivo 'ventas.csv'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

x = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas'].values, dtype=torch.float32).view(-1, 1)


with col2:
    st.subheader("Datos de Ventas Diarias")
    st.write(data.head())


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


model = NeuralNetwork(input_size=1, hidden_size=hidden_neurons, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if train_button:
    st.write("Entrenando la red neuronal...")
    loss_values = []
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

       
        loss_values.append(loss.item())
        
      
        progress_bar.progress((epoch + 1) / epochs)
        
       
        if (epoch + 1) % 100 == 0:
            st.write(f"Época {epoch + 1}/{epochs} - Error: {loss.item():.6f}")

    st.success("Entrenamiento finalizado con éxito")

    
    with col1:
        st.subheader("Evolución de la Pérdida")
        fig, ax = plt.subplots()
        ax.plot(range(epochs), loss_values, color='blue', label="Pérdida")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida")
        ax.legend()
        st.pyplot(fig)

    
    with col2:
        st.subheader("Clasificación de Ventas Diarias")
        with torch.no_grad():
            predicted = model(x).detach().numpy()

        fig2, ax2 = plt.subplots()
        ax2.scatter(data['dia'], data['ventas'], color='blue', label="Datos Reales")
        ax2.plot(data['dia'], predicted, color='red', label="Predicción")
        ax2.set_xlabel("Día del Mes")
        ax2.set_ylabel("Ventas")
        ax2.legend()
        st.pyplot(fig2)
