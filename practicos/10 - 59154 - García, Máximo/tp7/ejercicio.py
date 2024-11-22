import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Parámetros de Entrenamiento
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.01)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 1000)
hidden_neurons = st.sidebar.slider("Neuronas en la Capa Oculta", 1, 100, 10)
train_button = st.sidebar.button("Entrenar")

# Leer Datos
data = pd.read_csv("ventas.csv")
x_data = data['dia'].values.reshape(-1, 1)
y_data = data['ventas'].values.reshape(-1, 1)

# Normalizar Datos
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_data = scaler_x.fit_transform(x_data)
y_data = scaler_y.fit_transform(y_data)

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# Crear Red Neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Entrenar Red Neuronal
if train_button:
    model = NeuralNetwork(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    loss_values = []
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Guardar la pérdida y actualizar la barra de progreso
        loss_values.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        
        # Mostrar mensaje cada 100 epochs
        if (epoch + 1) % 100 == 0:
            st.write(f"Epoch {epoch + 1}/{epochs} - Error: {loss.item():.6f}")
    
    st.success("Entrenamiento exitoso")

    # Guardar Modelo
    torch.save(model.state_dict(), "modelo_ventas.pth")
    st.write("Modelo guardado como 'modelo_ventas.pth'")

    # Graficar la pérdida
    fig, ax = plt.subplots()
    ax.plot(loss_values, color="green", label="Pérdida")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.pyplot(fig)

    # Graficar Predicciones
    with torch.no_grad():
        predictions = model(x_train)
        predictions = scaler_y.inverse_transform(predictions.detach().numpy())
    
    # Graficar datos reales y predicciones
    fig, ax = plt.subplots()
    ax.scatter(data['dia'], data['ventas'], color="blue", label="Datos Reales")
    ax.plot(data['dia'], predictions, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
    # Profe, tuve muchos problemas para hacerlo andar en mi PC, con el tema de las dependecias y eso.
    # Tuve que hacerlo andar en la computadora de un compañero por las dudas.
    # profe sin querer copie mi codigo en la carpeta de un compañero,por error pero ya le deje el codigo que tenia y puese el mio en la carpeta correspondiente, mil disculpas
   