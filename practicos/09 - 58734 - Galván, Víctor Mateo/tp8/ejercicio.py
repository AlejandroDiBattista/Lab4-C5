import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = https://tp8-58734.streamlit.app

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58734')
        st.markdown('**Nombre:** victor mateo galvan')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()


# Título de la aplicación
st.title("Datos de Ventas de Todas las Sucursales")

# Cargar archivo CSV
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Asegurarnos de que las columnas Año y Mes sean numéricas
    data['Año'] = data['Año'].astype(int)
    data['Mes'] = data['Mes'].astype(int)

    # Seleccionar sucursal
    sucursal = st.sidebar.selectbox(
        "Seleccionar Sucursal",
        options=["Todas"] + list(data["Sucursal"].unique())
    )

    # Filtrar datos según la sucursal seleccionada
    if sucursal != "Todas":
        data = data[data["Sucursal"] == sucursal]

    # Agrupar por Producto
    productos = data["Producto"].unique()

    for producto in productos:
        # Filtrar datos por producto
        df_producto = data[data["Producto"] == producto]

        # Calcular Precio promedio, Margen promedio y Unidades vendidas
        precio_promedio = (df_producto["Ingreso_total"].sum() / 
                           df_producto["Unidades_vendidas"].sum())
        margen_promedio = ((df_producto["Ingreso_total"].sum() - 
                            df_producto["Costo_total"].sum()) / 
                           df_producto["Ingreso_total"].sum()) * 100
        unidades_vendidas = df_producto["Unidades_vendidas"].sum()

        # Mostrar estadísticas
        st.subheader(producto)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precio Promedio", f"${precio_promedio:.3f}")
        col2.metric("Margen Promedio", f"{margen_promedio:.2f}%")
        col3.metric("Unidades Vendidas", f"{unidades_vendidas:,}")

        # Crear columna de fecha usando el primer día del mes
        df_producto["Fecha"] = pd.to_datetime(df_producto["Año"].astype(str) + '-' + 
                                              df_producto["Mes"].astype(str) + '-01')

        # Ordenar por fecha para el gráfico
        df_producto = df_producto.sort_values("Fecha")

        # Gráfico de evolución de ventas
        fig, ax = plt.subplots()
        ax.plot(df_producto["Fecha"], df_producto["Unidades_vendidas"], label=producto, color="blue")

        # Agregar línea de tendencia
        X = np.array(range(len(df_producto))).reshape(-1, 1)
        y = df_producto["Unidades_vendidas"].values
        modelo = LinearRegression()
        modelo.fit(X, y)
        tendencia = modelo.predict(X)
        ax.plot(df_producto["Fecha"], tendencia, label="Tendencia", color="red", linestyle="--")

        # Configuración del gráfico
        ax.set_title("Evolución de Ventas Mensual")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades Vendidas")
        ax.legend()

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)
else:
    st.write("Por favor, sube un archivo CSV para comenzar.")
