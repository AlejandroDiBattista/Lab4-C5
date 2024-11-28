import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

## Función para mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58734')
        st.markdown('**Nombre:** Victor Mateo Galván')
        st.markdown('**Comisión:** C5')

# Título de la aplicación
st.title("Datos de Todas las Sucursales")

# Cargar archivo CSV desde el panel lateral
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file)

    # Asegurarnos de que las columnas Año y Mes sean numéricas
    data['Año'] = data['Año'].astype(int)
    data['Mes'] = data['Mes'].astype(int)

    # Seleccionar sucursal desde el panel lateral
    sucursal = st.sidebar.selectbox(
        "Seleccionar Sucursal",
        options=["Todas"] + list(data["Sucursal"].unique())
    )

    # Filtrar datos según la sucursal seleccionada
    if sucursal != "Todas":
        data = data[data["Sucursal"] == sucursal]

    # Agrupar por Producto
    productos = data["Producto"].unique()

    # Mostrar gráficos e información por producto
    for producto in productos:
        with st.container():
            st.subheader(producto)

            # Filtrar datos por producto
            datos_producto = data[data["Producto"] == producto]

            # Convertir a numérico y eliminar valores inválidos
            datos_producto["Ingreso_total"] = pd.to_numeric(datos_producto["Ingreso_total"], errors="coerce")
            datos_producto["Unidades_vendidas"] = pd.to_numeric(datos_producto["Unidades_vendidas"], errors="coerce")
            datos_producto["Costo_total"] = pd.to_numeric(datos_producto["Costo_total"], errors="coerce")

            # Eliminar filas con valores nulos en las columnas críticas
            datos_producto = datos_producto.dropna(subset=["Ingreso_total", "Unidades_vendidas", "Costo_total"])

            # Calcular Precio promedio basado en precios unitarios
            datos_producto["Precio_unitario"] = datos_producto["Ingreso_total"] / datos_producto["Unidades_vendidas"]
            precio_promedio = round(datos_producto["Precio_unitario"].mean(), 2)  # Redondear el precio promedio a 2 decimales

            # Calcular las ganancias promedio y el margen
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = round(datos_producto['Margen'].mean(), 2)  # Redondear el margen promedio a 2 decimales

            # Calcular Unidades vendidas totales
            unidades_vendidas = datos_producto["Unidades_vendidas"].sum()

            # Calcular variación porcentual del precio, margen y unidades vendidas
            datos_producto['Fecha'] = pd.to_datetime(datos_producto["Año"].astype(str) + '-' + 
                                                     datos_producto["Mes"].astype(str) + '-01')
            datos_producto = datos_producto.sort_values("Fecha")

            precio_anterior = datos_producto["Precio_unitario"].shift(1).mean()
            margen_anterior = datos_producto["Margen"].shift(1).mean()
            unidades_anterior = datos_producto["Unidades_vendidas"].shift(1).sum()

            variacion_precio = ((precio_promedio - precio_anterior) / precio_anterior) * 100 if precio_anterior else 0
            variacion_margen = ((margen_promedio - margen_anterior) / margen_anterior) * 100 if margen_anterior else 0
            variacion_unidades = ((unidades_vendidas - unidades_anterior) / unidades_anterior) * 100 if unidades_anterior else 0

            # Crear diseño con columnas
            col1, col2 = st.columns([1, 3])  # 25% para métricas, 75% para gráfico

            # Métricas a la izquierda
            with col1:
                st.metric(
                    "Precio Promedio",
                    f"${precio_promedio:,.2f}",  # El precio ya está redondeado a 2 decimales
                    f"{variacion_precio:,.2f}%",
                    delta_color="inverse" if variacion_precio < 0 else "normal"
                )
                st.metric(
                    "Margen Promedio",
                    f"{margen_promedio:,.2f}%",  # El margen ya está redondeado a 2 decimales
                    f"{variacion_margen:,.2f}%",
                    delta_color="inverse" if variacion_margen < 0 else "normal"
                )
                st.metric(
                    "Unidades Vendidas",
                    f"{unidades_vendidas:,}",
                    f"{variacion_unidades:,.2f}%",
                    delta_color="inverse" if variacion_unidades < 0 else "normal"
                )

            # Gráfico a la derecha
            with col2:
                fig, ax = plt.subplots()
                ax.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=producto, color="blue")

                # Agregar línea de tendencia
                X = np.array(range(len(datos_producto))).reshape(-1, 1)
                y = datos_producto["Unidades_vendidas"].values
                modelo = LinearRegression()
                modelo.fit(X, y)
                tendencia = modelo.predict(X)
                ax.plot(datos_producto["Fecha"], tendencia, label="Tendencia", color="red", linestyle="--")

                # Configuración del gráfico
                ax.set_title("Evolución de Ventas Mensual")
                ax.set_xlabel("Año-Mes")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()

                # Mostrar gráfico en Streamlit
                st.pyplot(fig)

else:
    # Mostrar información del alumno si no se han cargado datos
    mostrar_informacion_alumno()
    st.write("Por favor, sube un archivo CSV para comenzar.")
