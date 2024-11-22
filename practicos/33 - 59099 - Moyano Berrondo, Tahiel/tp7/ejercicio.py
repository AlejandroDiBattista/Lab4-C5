import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar y escalar los datos
df = pd.read_csv('ventas.csv')
escalador = MinMaxScaler()
df['ventas_escaladas'] = escalador.fit_transform(df[['ventas']])

# Definición de la red neuronal con una sola capa oculta
class PredictorVentas(nn.Module):
    def __init__(self, dim_entrada, dim_oculta, dim_salida):
        super(PredictorVentas, self).__init__()
        self.capa_oculta = nn.Linear(dim_entrada, dim_oculta)
        self.capa_salida = nn.Linear(dim_oculta, dim_salida)
    
    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))
        x = self.capa_salida(x)
        return x

# Configuración de la interfaz de Streamlit
st.title("Estimación de Ventas Diarias")
st.sidebar.header("Parámetros de Entrenamiento")

# Parámetros de entrada
tasa_aprendizaje = st.sidebar.number_input("Tasa de Aprendizaje", value=0.001, step=0.001, format="%.4f")
epocas = st.sidebar.number_input("Épocas", value=2000, step=100)
neuronas_ocultas = st.sidebar.number_input("Neuronas Capa Oculta", value=10, step=1)

# Entrenamiento al hacer clic en el botón
if st.sidebar.button("Entrenar"):
    # Preparación de datos
    X = torch.tensor(df[['dia']].values, dtype=torch.float32)
    y = torch.tensor(df[['ventas_escaladas']].values, dtype=torch.float32)
    
    # Inicialización del modelo y optimizador
    modelo = PredictorVentas(dim_entrada=1, dim_oculta=neuronas_ocultas, dim_salida=1)
    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    
    # Entrenamiento del modelo
    historial_perdida = []
    progreso = st.sidebar.progress(0)
    
    for epoca in range(epocas):
        modelo.train()
        optimizador.zero_grad()
        predicciones = modelo(X)
        perdida = criterio(predicciones, y)
        perdida.backward()
        optimizador.step()
        
        historial_perdida.append(perdida.item())
        progreso.progress((epoca + 1) / epocas)
        
    st.sidebar.success("Entrenamiento exitoso")
    st.sidebar.write(f"Época {epocas}/{epocas} - Error: {perdida.item():.6f}")

    # Gráfico de pérdida
    fig_perdida, ax_perdida = plt.subplots()
    ax_perdida.plot(historial_perdida, color="green")
    ax_perdida.set_xlabel("Época")
    ax_perdida.set_ylabel("Pérdida")
    ax_perdida.set_title("Pérdida")
    st.sidebar.pyplot(fig_perdida)
    
    # Predicción y desescalado
    modelo.eval()
    with torch.no_grad():
        predicciones_escaladas = modelo(X).numpy()
    predicciones_invertidas = escalador.inverse_transform(predicciones_escaladas)

    # Gráfico de ventas reales vs predicción
    fig, ax = plt.subplots()
    ax.plot(df['dia'], df['ventas'], 'o', color="blue", label="Datos Reales")
    ax.plot(df['dia'], predicciones_invertidas, color="red", label="Curva de Ajuste")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)