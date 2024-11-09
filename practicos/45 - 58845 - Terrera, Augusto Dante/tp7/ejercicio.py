import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def cargar_datos():
    df = pd.read_csv("ventas.csv")  
    return df

class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(1, neuronas_ocultas)
        self.fc2 = nn.Linear(neuronas_ocultas, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

st.sidebar.header("Parámetros de Entrenamiento")

col1, col2 = st.sidebar.columns(2)
tasa_aprendizaje = col1.number_input("Aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
epocas = col2.number_input("Repeticiones", min_value=10, max_value=10000, value=1000)

neuronas_ocultas = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10)
entrenar = st.sidebar.button("Entrenar")

df = cargar_datos()
st.write("### Estimación de Ventas Diarias")

def escalar_datos(data):
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

dias, min_dia, max_dia = escalar_datos(df['dia'].values)
ventas, min_ventas, max_ventas = escalar_datos(df['ventas'].values)

dias_tensor = torch.tensor(dias, dtype=torch.float32).view(-1, 1)
ventas_tensor = torch.tensor(ventas, dtype=torch.float32).view(-1, 1)

if entrenar:
    modelo = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)

    progreso = st.progress(0)
    perdidas = []

    for epoca in range(epocas):
        prediccion = modelo(dias_tensor)
        perdida = criterio(prediccion, ventas_tensor)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        perdidas.append(perdida.item())
        progreso.progress((epoca + 1) / epocas)

        if (epoca + 1) % (epocas // 10) == 0:
            st.sidebar.write(f"Epoch {epoca + 1}/{epocas} - Error: {perdida.item():.6f}")

    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Epoch {epocas}/{epocas} - Error: {perdida.item():.6f}")

    fig, ax = plt.subplots()
    ax.plot(perdidas, color="green", label="Pérdidas")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    st.sidebar.pyplot(fig) 

    with torch.no_grad():
        prediccion_final = modelo(dias_tensor).detach().numpy()
        prediccion_final = prediccion_final * (max_ventas - min_ventas) + min_ventas

    fig, ax = plt.subplots()
    ax.plot(df['dia'], df['ventas'], 'bo', label="Datos Reales")
    ax.plot(df['dia'], prediccion_final, 'r-', label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)
