import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias')


datos = pd.DataFrame({
    'dia': list(range(1, 31)),
    'ventas': [195, 169, 172, 178, 132, 123, 151, 127, 96, 110, 
               86, 82, 94, 60, 63, 76, 69, 98, 77, 71, 
               134, 107, 120, 99, 126, 150, 136, 179, 173, 194]
})

st.subheader("Datos de Ventas Diarias")
figura, grafico = plt.subplots()
grafico.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos Reales')
grafico.set_xlabel("Día del Mes")
grafico.set_ylabel("Ventas")
grafico.legend()
st.pyplot(figura)

st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.number_input("Aprendizaje", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
epocas = st.sidebar.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=100)
neuronas_ocultas = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10, step=1)
boton_entrenar = st.sidebar.button("Entrenar")

class RedVentas(nn.Module):
    def __init__(self, entrada, ocultas, salida):
        super(RedVentas, self).__init__()
        self.capa_oculta = nn.Linear(entrada, ocultas)
        self.capa_salida = nn.Linear(ocultas, salida)

    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))
        x = self.capa_salida(x)
        return x

escalador = MinMaxScaler()
datos['ventas_norm'] = escalador.fit_transform(datos[['ventas']])
x = torch.tensor(datos['dia'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(datos['ventas_norm'].values, dtype=torch.float32).view(-1, 1)

if boton_entrenar:
    modelo = RedVentas(entrada=1, ocultas=neuronas_ocultas, salida=1)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

    barra_progreso = st.sidebar.progress(0)
    perdidas = []

    for epoca in range(epocas):
        modelo.train()
        salida = modelo(x)
        perdida = criterio(salida, y)

        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        perdidas.append(perdida.item())
        if (epoca + 1) % (epocas // 10) == 0 or epoca == epocas - 1:
            barra_progreso.progress((epoca + 1) / epocas)
            st.sidebar.write(f"Época {epoca + 1}/{epocas} - Error: {perdida.item():.6f}")

    st.sidebar.success("Entrenamiento exitoso")

    st.sidebar.subheader("Evolución de la Función de Costo")
    figura, grafico = plt.subplots()
    grafico.plot(range(epocas), perdidas, 'g-', label='Pérdidas')
    grafico.set_xlabel("Épocas")
    grafico.set_ylabel("Pérdida")
    grafico.legend()
    st.sidebar.pyplot(figura)

    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(x).numpy()

    st.subheader("Predicción de Ventas Diarias")
    figura, grafico = plt.subplots()
    grafico.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos Reales')
    grafico.plot(datos['dia'], escalador.inverse_transform(predicciones), 'r-', label='Curva de Ajuste')
    grafico.set_xlabel("Día del Mes")
    grafico.set_ylabel("Ventas")
    grafico.legend()
    st.pyplot(figura)
