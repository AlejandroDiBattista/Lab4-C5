import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#Link: https://tp-n-8-58736.streamlit.app/
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

legajo = "58736"
nombre = "Lucas David Juarez Hindi"
comision = "C5"

st.sidebar.title("Datos del Empleado")
st.sidebar.write(f"**Legajo**: {legajo}")
st.sidebar.write(f"**Nombre**: {nombre}")
st.sidebar.write(f"**Comisión**: {comision}")

@st.cache_data
def cargar_datos(file):
    try:
        data = pd.read_csv(file)
        data["Unidades_vendidas"] = pd.to_numeric(data["Unidades_vendidas"], errors="coerce")
        data["Ingreso_total"] = pd.to_numeric(data["Ingreso_total"], errors="coerce")
        data["Costo_total"] = pd.to_numeric(data["Costo_total"], errors="coerce")
        return data
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

st.sidebar.title("Cargar archivo de datos")
file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if file:
    data = cargar_datos(file)

    st.sidebar.title("Seleccionar Sucursal")
    sucursales = ["Todas"] + sorted(data["Sucursal"].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Sucursal", sucursales)

    if sucursal_seleccionada == "Todas":
        data_filtrada = data
        st.title("Datos de Todas las Sucursales")
    else:
        data_filtrada = data[data["Sucursal"] == sucursal_seleccionada]
        st.title(f"Datos de la Sucursal: {sucursal_seleccionada}")

    productos = data_filtrada["Producto"].unique()

    for producto in productos:
        st.subheader(producto)

        data_producto = data_filtrada[data_filtrada["Producto"] == producto]

        total_unidades_vendidas = data_producto["Unidades_vendidas"].sum()
        ingreso_total = data_producto["Ingreso_total"].sum()
        costo_total = data_producto["Costo_total"].sum()

        if total_unidades_vendidas > 0:
            precio_promedio = ingreso_total / total_unidades_vendidas
        else:
            precio_promedio = 0

        if ingreso_total > 0:
            margen_promedio = ((ingreso_total - costo_total) / ingreso_total) * 100
        else:
            margen_promedio = 0

        variacion_precio = np.random.uniform(-20, 30)
        variacion_margen = np.random.uniform(-5, 5)
        variacion_unidades = np.random.uniform(-10, 10)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{variacion_precio:.2f}%")
        with col2:
            st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{variacion_margen:.2f}%")
        with col3:
            st.metric("Unidades Vendidas", f"{int(total_unidades_vendidas):,}", f"{variacion_unidades:.2f}%")

        data_producto["Año-Mes"] = pd.to_datetime(data_producto["Año"].astype(str) + "-" + data_producto["Mes"].astype(str))
        evolucion_ventas = data_producto.groupby("Año-Mes")["Unidades_vendidas"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=evolucion_ventas, x="Año-Mes", y="Unidades_vendidas", marker="o", ax=ax)

        evolucion_ventas["Fecha_num"] = (evolucion_ventas["Año-Mes"] - evolucion_ventas["Año-Mes"].min()) / np.timedelta64(1, 'D')
        coef = np.polyfit(evolucion_ventas["Fecha_num"], evolucion_ventas["Unidades_vendidas"], 1)
        poly = np.poly1d(coef)
        evolucion_ventas["Tendencia"] = poly(evolucion_ventas["Fecha_num"])
        ax.plot(evolucion_ventas["Año-Mes"], evolucion_ventas["Tendencia"], color='red', linestyle='--', label="Tendencia")

        ax.set_title(f"Evolución de Ventas Mensual: {producto}")
        ax.set_ylabel("Unidades Vendidas")
        ax.set_xlabel("Año-Mes")
        plt.xticks(rotation=45)
        ax.legend()

        st.pyplot(fig)
