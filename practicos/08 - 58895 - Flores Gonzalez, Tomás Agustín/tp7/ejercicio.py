import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


st.title('Estimación de Ventas Diarias')

class RedNeuronal(nn.Module):
    def __init__(self, entrada_oculta, tamaño_oculto, salida_oculta):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(entrada_oculta, tamaño_oculto)
        self.output = nn.Linear(tamaño_oculto, salida_oculta)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 1.0, 0.1, 0.01)
epocas = st.sidebar.slider("Épocas de Entrenamiento", 10, 10000, 100, 10)
neuronas_ocultas = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, 5, 1)
boton_entrenar = st.sidebar.button("Entrenar Modelo")

try:
    data = pd.read_csv('ventas.csv')
    dia = data['dia'].values.astype(np.float32)
    ventas = data['ventas'].values.astype(np.float32)
    dia_norm = (dia - dia.min()) / (dia.max() - dia.min())
    ventas_norm = (ventas - ventas.min()) / (ventas.max() - ventas.min())
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

def graficar_predicciones(dia, ventas, modelo, entradas, titulo):
    fig, ax = plt.subplots()
    ax.plot(dia, ventas, 'bo', label="Datos Reales")
    with torch.no_grad():
        predecido = modelo(entradas).numpy()
    predecido_rescalado = predecido * (ventas.max() - ventas.min()) + ventas.min()
    ax.plot(dia, predecido_rescalado, 'r-', label="Curva de Ajuste")
    ax.set_title(titulo)
    ax.set_xlabel('Día del Mes')
    ax.set_ylabel('Ventas')
    ax.legend()
    st.pyplot(fig)

def entrenar_red_neuronal(entradas, etiquetas, modelo, epocas, criterio, optimizador):
    valores_perdida = []
    barra_progreso = st.sidebar.empty()
    mensaje_epoch = st.sidebar.empty()

    for epoca in range(epocas):
        salidas = modelo(entradas)
        perdida = criterio(salidas, etiquetas)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        valores_perdida.append(perdida.item())
        if epocas >= 100 and (epoca + 1) % (epocas // 100) == 0:
            barra_progreso.progress((epoca + 1) / epocas)
        mensaje_epoch.markdown(f"**Época {epoca + 1}/{epocas} - Pérdida: {perdida.item():.6f}**")

    st.sidebar.success("Entrenamiento completo")
    return valores_perdida

if boton_entrenar:
    entradas = torch.from_numpy(dia_norm.reshape(-1, 1))
    etiquetas = torch.from_numpy(ventas_norm.reshape(-1, 1))
    modelo = RedNeuronal(entrada_oculta=1, tamaño_oculto=neuronas_ocultas, salida_oculta=1)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

    valores_perdida = entrenar_red_neuronal(entradas, etiquetas, modelo, epocas, criterio, optimizador)

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(range(1, epocas + 1), valores_perdida, 'b-', label="Pérdidas")
    ax_loss.set_xlabel('Épocas')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.legend()
    st.pyplot(fig_loss)

    torch.save(modelo.state_dict(), 'modelo_ventas.pth')
    st.sidebar.info("Modelo guardado como 'modelo_ventas.pth'")

    graficar_predicciones(dia, ventas, modelo, entradas, "Curva de Ajuste de Ventas")

st.write("Ajusta los parámetros en el panel lateral y entrena el modelo para estimar las ventas diarias.")