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
    
st.title('Estimación de Ventas Diarias')

data = pd.read_csv('ventas.csv')

ventas = data['ventas'].values
ventas_min = np.min(ventas)
ventas_max = np.max(ventas)
ventas_normalizadas = (ventas - ventas_min) / (ventas_max - ventas_min)
data['ventas_normalizadas'] = ventas_normalizadas

x = torch.tensor(data['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['ventas_normalizadas'].values, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons):
        super(SimpleNN, self).__init__()
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
st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurones en la capa oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

if train_button:
    model = SimpleNN(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    progress_bar = st.progress(0)
    loss_history = []

    for epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y)
        
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
    loss_history.append(loss.item())
    progress_bar.progress((epoch + 1) / epochs)
    
    st.success("Entrenamiento finalizado")
    
    torch.save(model.state_dict(), 'modelo_ventas.pth')
    
    
    fig, ax = plt.subplots()
    ax.plot(loss_history, label="Pérdida")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Pérdida")
    ax.set_title("Evolución de la función de costo")
    st.sidebar.pyplot(fig)  

    with torch.no_grad():
        predictions = model(x).numpy()
        predictions = predictions * (ventas_max - ventas_min) + ventas_min  
        actual_sales = data['ventas'].values  

    fig, ax = plt.subplots()
    ax.plot(data['dia'], actual_sales, label="Ventas Reales", color="blue")
    ax.plot(data['dia'], predictions, label="Predicción de la Red", color="red")
    ax.set_xlabel("Día del mes")
    ax.set_ylabel("Ventas")
    ax.set_title("Ventas diarias y Predicción de la Red Neuronal")
    ax.legend()
    st.pyplot(fig)
