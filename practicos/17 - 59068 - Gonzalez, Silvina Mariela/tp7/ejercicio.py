import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

# Crear Red Neuronal
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

# Leer Datos
data = pd.read_csv('ventas.csv')

# Normalizar Datos
dia = data['dia'].values.astype(np.float32)
ventas = data['ventas'].values.astype(np.float32)
dia_norm = (dia - dia.min()) / (dia.max() - dia.min())
ventas_norm = (ventas - ventas.min()) / (ventas.max() - ventas.min())

tasa_aprendizaje = st.session_state.get("tasa_aprendizaje", 0.1)
st.sidebar.write("Aprendizaje:")
cols = st.sidebar.columns([1, 2, 1])
if cols[0].button("➖", key="lr_decrement"):
    tasa_aprendizaje = max(0.0, tasa_aprendizaje - 0.01)
cols[1].write(f"{tasa_aprendizaje:.2f}")
if cols[2].button("➕", key="lr_increment"):
    tasa_aprendizaje = min(1.0, tasa_aprendizaje + 0.01)
st.session_state["tasa_aprendizaje"] = tasa_aprendizaje

epocas = st.session_state.get("epocas", 100)
st.sidebar.write("Repeticiones:")
cols = st.sidebar.columns([1, 2, 1])
if cols[0].button("➖", key="epocas_decrement"):
    epocas = max(10, epocas - 10)
cols[1].write(f"{epocas}")
if cols[2].button("➕", key="epocas_increment"):
    epocas = min(10000, epocas + 10)
st.session_state["epocas"] = epocas

neuronas_ocultas = st.session_state.get("neuronas_ocultas", 5)
st.sidebar.write("Neuronas Capa Oculta:")
cols = st.sidebar.columns([1, 2, 1])
if cols[0].button("➖", key="neuronas_ocultas_decrement"):
    neuronas_ocultas = max(1, neuronas_ocultas - 1)
cols[1].write(f"{neuronas_ocultas}")
if cols[2].button("➕", key="neuronas_ocultas_increment"):
    neuronas_ocultas = min(100, neuronas_ocultas + 1)
st.session_state["neuronas_ocultas"] = neuronas_ocultas

boton_entrenar = st.sidebar.button("Entrenar")
barra_progreso = st.sidebar.empty()
mensaje_epoch = st.sidebar.empty()
mensaje_exitoso  = st.sidebar.empty()
grafico_perdida = st.sidebar.empty()

# Entrenar Red Neuronal
if boton_entrenar:
    entradas = torch.from_numpy(dia_norm.reshape(-1, 1))
    etiquetas = torch.from_numpy(ventas_norm.reshape(-1, 1))

    modelo = RedNeuronal(entrada_oculta=1, tamaño_oculto=neuronas_ocultas, salida_oculta=1)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

    valores_perdida = []
    barra_progreso.progress(0)
    
    for epoca in range(epocas):
        salidas = modelo(entradas)
        perdida = criterio(salidas, etiquetas)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        valores_perdida.append(perdida.item())

        if epocas >= 100:
            if (epoca + 1) % (epocas // 100) == 0:
                barra_progreso.progress((epoca + 1) / epocas)
        mensaje_epoch.markdown(f"**Epoch {epoca + 1}/{epocas} - Error: {perdida.item():.6f}**")
    mensaje_exitoso.success("Entrenamiento exitoso")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(range(1, epocas + 1), valores_perdida, 'b-', label="Pérdidas")
    ax_loss.set_xlabel('Épocas')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.legend()
    grafico_perdida.pyplot(fig_loss)

    # Guardar Modelo
    torch.save(modelo.state_dict(), 'modelo_ventas.pth')

    # Graficar Predicciones
    fig, ax = plt.subplots()
    ax.plot(dia, ventas, 'bo', label="Datos Reales")
    with torch.no_grad():
        predecido = modelo(entradas).numpy()
    predecido_rescalado = predecido * (ventas.max() - ventas.min()) + ventas.min()
    ax.plot(dia, predecido_rescalado, 'r-', label="Curva de Ajuste")
    ax.set_xlabel('Día del Mes')
    ax.set_ylabel('Ventas')
    ax.legend()
    st.pyplot(fig)