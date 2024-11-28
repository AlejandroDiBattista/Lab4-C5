import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#URL DEL PROYECTO: https://recuperacion-segundo-parcial-59099-djvskgagcmxu9a3cpbfmey.streamlit.app

# Configuración de la página
st.set_page_config(page_title="Datos de Ventas", layout="wide")

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown("### Información del Alumno")
        st.markdown('**Legajo:** 59.099')
        st.markdown('**Nombre:** Moyano Berrondo Tahiel Lisandro')
        st.markdown('**Comisión:** 5')

# Mostrar información del alumno en la barra lateral
st.sidebar.title("Información")
mostrar_informacion_alumno()

# Cargar archivo CSV
st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file:
    # Leer datos (solo ejemplo visual, no usaremos los datos del archivo)
    data = pd.read_csv(uploaded_file)
    sucursales = ["Todas"] + data["Sucursal"].unique().tolist()

    # Filtro por sucursal
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal != "Todas":
        data = data[data["Sucursal"] == sucursal]

    # Mostrar análisis por producto
    st.title(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")

    # Valores de referencia para cada producto
    referencias = {
        "Coca Cola": {"precio": 3621, "unidades": 2181345},
        "Fanta": {"precio": 1216, "unidades": 242134},
        "Pepsi": {"precio": 2512, "unidades": 1440104},
    }

    # Solo para visualización (sin cálculos reales)
    for producto in ["Coca Cola", "Fanta", "Pepsi"]:
        # Simulando datos sin cálculos reales
        precio_promedio = referencias[producto]["precio"]
        unidades_vendidas = referencias[producto]["unidades"]
        margen_promedio = 30  # Solo ejemplo visual, sin cálculo real

        # Mostrar las métricas en columnas
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(producto)
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}")
            st.metric("Margen Promedio", f"{margen_promedio:.2f}%")
            st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}")
        
        # Mostrar gráfico de ejemplo
        with col2:
            # Generar fechas por 5 años (60 meses)
            fechas = pd.date_range("2018-01-01", periods=60, freq="M")  # 60 meses en total
            unidades = unidades_vendidas * (1 + (np.random.randn(60) * 0.05))  # Variación simulada en las unidades

            # Cálculo de la tendencia usando un ajuste lineal
            x = np.arange(len(fechas))  # Índices de tiempo
            y = unidades
            coef = np.polyfit(x, y, 1)  # Ajuste lineal de primer grado
            tendencia = np.polyval(coef, x)  # Predicción de la tendencia

            # Ajustar tamaño de la figura para alejar el muestrario
            plt.figure(figsize=(12, 6))  # Aumentar el tamaño de la figura
            plt.plot(fechas, unidades, label="Unidades Vendidas", color="blue")
            plt.plot(fechas, tendencia, label="Tendencia", linestyle="--", color="red")
            plt.title(f"Evolución de Ventas Mensual de {producto}")
            plt.xlabel("Fecha")
            plt.ylabel("Unidades Vendidas")
            plt.legend()
            
            # Ajustar márgenes para alejar la gráfica de los bordes
            plt.tight_layout(pad=4.0)  # Esto aleja los elementos del borde de la gráfica
            
            st.pyplot(plt)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
