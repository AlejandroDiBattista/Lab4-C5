import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.title('Estimación de Ventas Diarias')

# Cargar y visualizar los datos
data = pd.read_csv('ventas.csv')  # Asegúrate de que el archivo está en el mismo directorio o especifica la ruta completa
X = data[['dia']].values.astype(float)
y = data[['ventas']].values.astype(float)

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y)

# Configuración de los parámetros de la red neuronal en la barra lateral
st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.001, 1.0, 0.01)
epochs = st.sidebar.slider("Repeticiones", 10, 10000, 1000)
hidden_neurons = st.sidebar.slider("Neuronas Capa Oculta", 1, 100, 10)

# Crear la red neuronal
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Inicializar la red, el criterio de pérdida y el optimizador
model = NeuralNet(input_size=1, hidden_size=hidden_neurons, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convertir datos a tensores
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

# Botón de entrenamiento
if st.sidebar.button("Entrenar"):
    loss_values = []
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Guardar los valores de la pérdida para graficar
        loss_values.append(loss.item())
        
        # Mostrar barra de progreso
        st.sidebar.progress((epoch + 1) / epochs)

    # Mensaje de éxito
    st.sidebar.success("Entrenamiento exitoso")

    # Graficar el valor de la pérdida
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(range(epochs), loss_values, 'g')
    ax_loss.set_title("Pérdida")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Pérdida")
    st.sidebar.pyplot(fig_loss)
    
    # Predicciones
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Desnormalizar para mostrar los resultados originales
    X_original = scaler_X.inverse_transform(X_normalized)
    y_original = scaler_y.inverse_transform(y_normalized)
    predictions_original = scaler_y.inverse_transform(predictions)

    # Graficar los datos y las predicciones
    fig, ax = plt.subplots()
    ax.plot(X_original, y_original, 'bo', label="Datos Reales")
    ax.plot(X_original, predictions_original, 'r-', label="Curva de Ajuste")
    ax.set_title("Estimación de Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
