import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv("ventas.csv")
    if "dia" in data.columns:
        data = data.rename(columns={"dia": "día"})
    return data

data = load_data()

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

st.sidebar.title("Parámetros de Entrenamiento")
learning_rate = st.sidebar.number_input("Aprendizaje", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
epochs = st.sidebar.slider("Repeticiones", 10, 1000, 100)
hidden_neurons = st.sidebar.slider("Neuronas Capa Oculta", 1, 100, 10)
train_button = st.sidebar.button("Entrenar")

st.title("Estimación de Ventas Diarias")

if train_button:
    st.write("Entrenando la red neuronal...")

    X = torch.tensor(data["día"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(data["ventas"].values, dtype=torch.float32).view(-1, 1)


    model = SimpleNN(input_size=1, hidden_size=hidden_neurons, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    loss_history = []
    loss_table = [] 
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        progress_bar.progress((epoch + 1) / epochs)
        
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            loss_table.append({"Época": epoch + 1, "Pérdida": loss.item()})

    st.success("Entrenamiento exitoso")

    
    st.write("**Evolución de la función de costo:**")
    fig, ax = plt.subplots()
    ax.plot(range(epochs), loss_history, label="Pérdidas", color="green")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (MSE)")
    st.pyplot(fig)

    st.write("**Estimación de Ventas Diarias:**")
    with torch.no_grad():
        predictions = model(X).detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(data["día"], data["ventas"], label="Datos Reales", color="blue")
    ax.plot(data["día"], predictions, label="Curva de Ajuste", color="red")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
