# app.py - Aplicación de datos de ventas con Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página (DEBE IR AL INICIO)
st.set_page_config(page_title="Datos de Ventas", layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = https://lab4-59099-parcial-b77sxoyydhtaep42mykpzq.streamlit.app'

# Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown("### Información del Alumno")
        st.markdown('**Legajo:** 59.099')
        st.markdown('**Nombre:** Moyano Berrondo Tahiel Lisandro')
        st.markdown('**Comisión:** C5')

# Mostrar información del alumno en la barra lateral
st.sidebar.title("Información")
mostrar_informacion_alumno()

# Cargar archivo CSV
st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
if uploaded_file:
    # Leer datos
    data = pd.read_csv(uploaded_file)
    sucursales = ["Todas"] + data["Sucursal"].unique().tolist()

    # Filtro por sucursal
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal != "Todas":
        data = data[data["Sucursal"] == sucursal]

    # Mostrar análisis por producto
    st.title(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")

    for producto in data["Producto"].unique():
        producto_data = data[data["Producto"] == producto]
        precio_promedio = (producto_data["Ingreso_total"].sum() / producto_data["Unidades_vendidas"].sum())
        margen_promedio = ((producto_data["Ingreso_total"].sum() - producto_data["Costo_total"].sum()) / 
                           producto_data["Ingreso_total"].sum()) * 100
        unidades_vendidas = producto_data["Unidades_vendidas"].sum()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(producto)
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}")
            st.metric("Margen Promedio", f"{margen_promedio:.2f}%")
            st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}")

        with col2:
            # Gráfico de evolución
            producto_data_grouped = producto_data.groupby(["Año", "Mes"]).sum().reset_index()

            # Crear la columna "Fecha" correctamente
            producto_data_grouped["Fecha"] = pd.to_datetime(
                producto_data_grouped["Año"].astype(str) + "-" + producto_data_grouped["Mes"].astype(str) + "-01",
                errors="coerce"
            )
            
            # Validar fechas no válidas
            producto_data_grouped = producto_data_grouped.dropna(subset=["Fecha"])
            
            # Ordenar los datos por fecha
            producto_data_grouped = producto_data_grouped.sort_values("Fecha")

            # Crear el gráfico
            plt.figure(figsize=(10, 4))
            plt.plot(producto_data_grouped["Fecha"], producto_data_grouped["Unidades_vendidas"], label="Unidades Vendidas")
            plt.plot(
                producto_data_grouped["Fecha"],
                np.poly1d(np.polyfit(
                    np.arange(len(producto_data_grouped["Fecha"])),
                    producto_data_grouped["Unidades_vendidas"], 1
                ))(np.arange(len(producto_data_grouped["Fecha"]))),
                label="Tendencia", linestyle="--"
            )
            plt.title("Evolución de Ventas Mensual")
            plt.xlabel("Fecha")
            plt.ylabel("Unidades Vendidas")
            plt.legend()
            st.pyplot(plt)

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
