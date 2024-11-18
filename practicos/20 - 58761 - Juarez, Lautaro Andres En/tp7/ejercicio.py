import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube el archivo de datos (ventas.csv)", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.write(df.head())

    # Normalizar datos
    st.write("Normalizando los datos...")
    scaler = MinMaxScaler()
    datos_normalizados = scaler.fit_transform(df.values)
    df_normalizado = pd.DataFrame(datos_normalizados, columns=df.columns)
    st.write("Datos normalizados:")
    st.write(df_normalizado.head())

    # Dividir datos en características (X) y etiquetas (y)
    X = df_normalizado.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = df_normalizado.iloc[:, -1].values   # Última columna
    y = y.reshape(-1, 1)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir datos a tensores de PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Definir la red neuronal
    class RedNeuronal(nn.Module):
        def __init__(self, input_size):
            super(RedNeuronal, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)  # Capa oculta
            self.fc2 = nn.Linear(32, 16)         # Capa oculta
            self.fc3 = nn.Linear(16, 1)          # Capa de salida
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Crear la red
    input_size = X_train.shape[1]
    model = RedNeuronal(input_size)
    criterion = nn.MSELoss()  # Función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimizador

    # Entrenar la red
    st.write("Entrenando la red neuronal...")
    epochs = 500
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mostrar progreso cada 50 epochs
        if (epoch + 1) % 50 == 0:
            st.write(f"Epoch {epoch+1}/{epochs}, Pérdida: {loss.item()}")

    # Guardar el modelo
    torch.save(model.state_dict(), "modelo_ventas.pth")
    st.write("Modelo guardado como `modelo_ventas.pth`.")

    # Graficar predicciones
    st.write("Generando predicciones...")
    with torch.no_grad():
        y_pred_test = model(X_test)
        y_pred_test = y_pred_test.numpy()
        y_test_np = y_test.numpy()

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_np, label="Valores reales")
    plt.plot(y_pred_test, label="Predicciones", linestyle="dashed")
    plt.legend()
    plt.title("Comparación de Valores Reales vs Predicciones")
    plt.xlabel("Índice")
    plt.ylabel("Ventas (normalizadas)")
    st.pyplot(plt)
