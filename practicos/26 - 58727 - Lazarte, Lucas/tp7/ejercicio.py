import os
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar y escalar los datos
def cargar_datos():
    """Carga los datos de ventas desde un archivo CSV y los escala."""
    df = pd.read_csv('ventas.csv')
    scaler = MinMaxScaler()
    df['ventas_scaled'] = scaler.fit_transform(df[['ventas']])
    return df, scaler

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

# Función para entrenar el modelo
def entrenar_modelo(model, optimizer, criterion, X, y, epochs, progress_bar):
    """Entrena el modelo, actualiza la barra de progreso y devuelve el historial de pérdidas."""
    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
    return loss_history, loss.item()

# Configuración de la interfaz de Streamlit
def mostrar_interfaz():
    """Muestra los controles de la interfaz de usuario en Streamlit."""
    st.title("Estimación de Ventas Diarias ")
    st.sidebar.header("Parámetros de Entrenamiento")

    # Parámetros de entrada
    learning_rate = st.sidebar.number_input("Aprendizaje", value=0.01, step=0.001, format="%.4f", min_value=0.0, max_value=1.0)
    epochs = st.sidebar.number_input("Repeticiones", value=1000, step=100, min_value=10, max_value=10000)
    hidden_neurons = st.sidebar.number_input("Neuronas en la Capa Oculta", value=10, step=1, min_value=1, max_value=100)

    return learning_rate, epochs, hidden_neurons

# Configuración y entrenamiento al hacer clic en "Entrenar"
def main():
    learning_rate, epochs, hidden_neurons = mostrar_interfaz()

    # Cargar datos
    df, scaler = cargar_datos()

    # Preparar los datos para el entrenamiento
    X = torch.tensor(df[['dia']].values, dtype=torch.float32)
    y = torch.tensor(df[['ventas_scaled']].values, dtype=torch.float32)

    # Inicialización del modelo y optimizador
    model = SalesPredictor(input_dim=1, hidden_dim=hidden_neurons, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Botón para entrenar el modelo
    if st.sidebar.button("Entrenar"):
        # Barra de progreso
        progress_bar = st.sidebar.progress(0)

        # Entrenamiento del modelo
        loss_history, final_loss = entrenar_modelo(model, optimizer, criterion, X, y, epochs, progress_bar)

        # Visualización del entrenamiento
        st.sidebar.success(f"Entrenamiento completado con éxito")
        st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {final_loss:.6f}")

        # Gráfico de la pérdida
        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
        ax_loss.plot(loss_history, color="green")
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Pérdida")
        ax_loss.set_title("Evolución de la Pérdida Durante el Entrenamiento")
        st.sidebar.pyplot(fig_loss)

        # Predicción y desescalado de los datos
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X).numpy()
        pred_inversed = scaler.inverse_transform(pred_scaled)

        # Gráfico de ventas reales vs predicción
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['dia'], df['ventas'], color="blue", label="Datos Reales")
        ax.plot(df['dia'], pred_inversed, color="red", label="Curva de Ajuste")
        ax.set_xlabel("Día del Mes")
        ax.set_ylabel("Ventas")
        ax.set_title("Predicción de Ventas Diarias  ")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

