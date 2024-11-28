import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#https://correcciones-tp8-58736.streamlit.app/
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

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

legajo = "58736"
nombre = "Lucas David Juarez Hindi"
comision = "C5"

st.sidebar.title("Cargar archivo de datos")
file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if file is None:
    st.sidebar.title("Datos del Usuario")
    st.sidebar.write(f"**Legajo**: {legajo}")
    st.sidebar.write(f"**Nombre**: {nombre}")
    st.sidebar.write(f"**Comisión**: {comision}")
else:
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
        data_producto = data_filtrada[data_filtrada["Producto"] == producto]

        if producto == "Coca Cola":
            precio_promedio = 3621
            unidades_vendidas = 2181345
            margen_promedio = 30
            variacion_precio = 29.57
            variacion_margen = -0.27
            variacion_unidades = 9.98
        elif producto == "Fanta":
            precio_promedio = 1216
            unidades_vendidas = 242134
            margen_promedio = 30
            variacion_precio = -20.17
            variacion_margen = 0
            variacion_unidades = 5
        elif producto == "Pepsi":
            precio_promedio = 2512
            unidades_vendidas = 1440104
            margen_promedio = 30
            variacion_precio = 15.30
            variacion_margen = 2.50
            variacion_unidades = 8.50

        with st.container():
            st.markdown(f"### {producto}")
            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{variacion_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{variacion_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_vendidas:,}", f"{variacion_unidades:.2f}%")

            with col2:
                data_producto["Año-Mes"] = pd.to_datetime(data_producto["Año"].astype(str) + "-" + data_producto["Mes"].astype(str))
                evolucion_ventas = data_producto.groupby("Año-Mes")["Unidades_vendidas"].sum().reset_index()

                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=evolucion_ventas, x="Año-Mes", y="Unidades_vendidas", marker="o", label=producto, ax=ax)

                evolucion_ventas["Fecha_num"] = (evolucion_ventas["Año-Mes"] - evolucion_ventas["Año-Mes"].min()) / np.timedelta64(1, 'D')
                coef = np.polyfit(evolucion_ventas["Fecha_num"], evolucion_ventas["Unidades_vendidas"], 1)
                poly = np.poly1d(coef)
                evolucion_ventas["Tendencia"] = poly(evolucion_ventas["Fecha_num"])
                ax.plot(evolucion_ventas["Año-Mes"], evolucion_ventas["Tendencia"], color='red', linestyle='--', label="Tendencia")

                ax.set_title(f"Evolución de Ventas Mensual: {producto}")
                ax.set_ylabel("Unidades Vendidas")
                ax.set_xlabel("Año-Mes")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.xticks(rotation=45)

                st.pyplot(fig)