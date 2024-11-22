import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1. Crear Red Neuronal
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# 2. Leer Datos
@st.cache_data
def load_data():
    data = pd.read_csv("ventas.csv")
    return data

# 3. Normalizar Datos
def normalize_data(data):
    x = torch.tensor(data["dia"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(data["ventas"].values, dtype=torch.float32).view(-1, 1)
    return x, y

# 4. Entrenar Red Neuronal
def train_model(model, criterion, optimizer, data_loader, epochs, progress_bar):
    losses = []
    for epoch in range(epochs):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
    return losses

# 5. Guardar Modelo
def save_model(model, path="modelo_ventas.pth"):
    torch.save(model.state_dict(), path)

# 6. Graficar Resultados
def plot_results(data, predictions, losses):
    # Gráfico 1: Evolución de la pérdida (en la sidebar)
    with st.sidebar:
        st.write("### Evolución de la Pérdida")
        fig1, ax1 = plt.subplots()
        ax1.plot(range(len(losses)), losses, label="Pérdidas", color="green")
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("Pérdida")
        ax1.legend()
        st.pyplot(fig1)

    # Gráfico 2: Predicciones y datos reales (en el cuerpo principal)
    st.write("### Predicciones vs. Datos Reales")
    fig2, ax2 = plt.subplots()
    ax2.scatter(data["dia"], data["ventas"], label="Datos Reales", color="blue")
    ax2.plot(data["dia"], predictions, label="Curva de Ajuste", color="red")
    ax2.set_xlabel("Día del Mes")
    ax2.set_ylabel("Ventas")
    ax2.legend()
    st.pyplot(fig2)

# 7. Aplicación Streamlit
def main():
    st.title("Estimación de Ventas Diarias con Redes Neuronales")
    st.sidebar.header("Parámetros de Entrenamiento")

    # Parámetros de entrada
    learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.001, 1.0, 0.01, 0.001)
    epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100, 10)
    hidden_size = st.sidebar.slider("Neuronas en la capa oculta", 1, 100, 10, 1)

    if st.sidebar.button("Entrenar"):
        data = load_data()
        x, y = normalize_data(data)
        dataset = TensorDataset(x, y)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = SimpleNN(input_size=1, hidden_size=hidden_size, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Barra de progreso
        progress_bar = st.progress(0)
        losses = train_model(model, criterion, optimizer, data_loader, epochs, progress_bar)

        st.success("Entrenamiento exitoso")

        # Guardar modelo
        save_model(model)

        # Predicciones
        with torch.no_grad():
            predictions = model(x).numpy()

        # Graficar resultados
        plot_results(data, predictions, losses)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
