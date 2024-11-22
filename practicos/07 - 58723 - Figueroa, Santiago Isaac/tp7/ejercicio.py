import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## Crear Red Neuronal
class RedVentas(nn.Module):
        def __init__(self, neuronas_ocultas):
            super(RedVentas, self).__init__()
            self.oculta = nn.Linear(1, neuronas_ocultas)
            self.salida = nn.Linear(neuronas_ocultas, 1)

        def forward(self, x):
            x = torch.relu(self.oculta(x))
            x = self.salida(x)
            return x
## Leer Datos
archivo_subido = st.file_uploader("Subir archivo ventas.csv", type="csv")
if archivo_subido:
    datos = pd.read_csv(archivo_subido)
    
## Normalizar Datos
    dias = datos['dia'].values.reshape(-1, 1).astype(np.float32)
    ventas = datos['ventas'].values.reshape(-1, 1).astype(np.float32)
    dias_norm = (dias - dias.min()) / (dias.max() - dias.min())
    ventas_norm = (ventas - ventas.min()) / (ventas.max() - ventas.min())
# Interfaz de Streamlit
    st.title('Estimación de Ventas Diarias')

    # Parámetros en la barra lateral
    st.sidebar.header("Parámetros de Entrenamiento")
    tasa_aprendizaje = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1, 0.01)
    epocas = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
    neuronas_ocultas = st.sidebar.slider("Neuronas en Capa Oculta", 1, 100, 5)
    boton_entrenar = st.sidebar.button("Entrenar")

    if boton_entrenar:
        # Inicializar modelo, función de pérdida y optimizador
        modelo = RedVentas(neuronas_ocultas)
        criterio = nn.MSELoss()
        optimizador = optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

        # Convertir los datos a tensores
        x_entrenamiento = torch.tensor(dias_norm, dtype=torch.float32)
        y_entrenamiento = torch.tensor(ventas_norm, dtype=torch.float32)
        perdidas = []
        
        barra_progreso = st.progress(0)
        
        for epoca in range(epocas):
            modelo.train()
            optimizador.zero_grad()
            salidas = modelo(x_entrenamiento)
            perdida = criterio(salidas, y_entrenamiento)
            perdida.backward()
            optimizador.step()
            perdidas.append(perdida.item())

            # Actualizar barra de progreso
            barra_progreso.progress((epoca + 1) / epocas)
        
        st.success("Entrenamiento exitoso")

        # Graficar pérdida a través de las épocas
        fig, ax = plt.subplots()
        ax.plot(range(epocas), perdidas, label='Pérdida', color='green')
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida")
        ax.legend()
        st.pyplot(fig)

        # Predecir y graficar resultados
        modelo.eval()
        with torch.no_grad():
            predicciones = modelo(x_entrenamiento).numpy()
        
        # Desnormalizar para graficar
        dias = dias.reshape(-1)  # Convertir a 1D para graficar
        ventas = ventas.reshape(-1)
        predicciones = predicciones * (ventas.max() - ventas.min()) + ventas.min()

        # Gráfico de ventas reales vs predicciones
        fig, ax = plt.subplots()
        ax.scatter(dias, ventas, label="Datos Reales", color="blue")
        ax.plot(dias, predicciones, label="Curva de Ajuste", color="red")
        ax.set_xlabel("Día del Mes")
        ax.set_ylabel("Ventas")
        ax.legend()
        st.pyplot(fig)