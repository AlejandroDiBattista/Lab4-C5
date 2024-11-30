import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://marianotp8c5.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.488')
        st.markdown('**Nombre:** Mariano E. Gonzalez')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()

# ======== Funciones auxiliares ========

def cargar_datos(uploaded_file):
    """Carga el archivo CSV y valida que tenga las columnas necesarias."""
    columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
    try:
        datos = pd.read_csv(uploaded_file)
        if not all(col in datos.columns for col in columnas_requeridas):
            st.error(f"El archivo debe contener las columnas: {', '.join(columnas_requeridas)}")
            return None
        return datos
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

def calcular_variaciones(df, producto):
    """Calcula las variaciones de precio, margen y unidades para un producto."""
    datos_producto = df[df['Producto'] == producto]  # Filtrar por producto
    datos_producto["Fecha"] = pd.to_datetime(
        datos_producto["Año"].astype(str) + '-' + datos_producto["Mes"].astype(str) + '-01',
        errors="coerce"
    )
    datos_producto.sort_values("Fecha", inplace=True)

    # Calcular variaciones
    precio_promedio_anual = datos_producto.groupby('Año').apply(
        lambda x: x["Ingreso_total"].sum() / x["Unidades_vendidas"].sum()
    )
    margen_promedio_anual = datos_producto.groupby('Año').apply(
        lambda x: ((x["Ingreso_total"].sum() - x["Costo_total"].sum()) / x["Ingreso_total"].sum()) * 100
    )
    unidades_vendidas_anual = datos_producto.groupby('Año')["Unidades_vendidas"].sum()

    return {
        "precio": precio_promedio_anual.pct_change().mean() * 100,
        "margen": margen_promedio_anual.pct_change().mean() * 100,
        "unidades": unidades_vendidas_anual.pct_change().mean() * 100,
    }

def generar_grafico(ventas_mensuales, producto):
    """Genera un gráfico con la evolución de ventas y la tendencia."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gráfico de unidades vendidas
    ax.plot(ventas_mensuales["Fecha"], ventas_mensuales["Unidades_vendidas"], label="Unidades Vendidas", color="blue")

    # Línea de tendencia
    if len(ventas_mensuales) > 1:
        x = np.arange(len(ventas_mensuales)).reshape(-1, 1)
        y = ventas_mensuales["Unidades_vendidas"].values.reshape(-1, 1)
        modelo = LinearRegression()
        modelo.fit(x, y)
        tendencia = modelo.predict(x)
        ax.plot(ventas_mensuales["Fecha"], tendencia, color="red", linestyle="--", label="Tendencia")

    # Configuraciones del gráfico
    ax.set_title(f"Evolución de Ventas Mensuales - {producto}", fontsize=14)
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Unidades Vendidas", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Formatear eje X
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Líneas divisorias mensuales
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Líneas divisorias anuales
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Mostrar solo años en las etiquetas principales
    ax.tick_params(axis="x", rotation=45)  # Rotar las etiquetas para mejor legibilidad

    # Mostrar líneas verticales menores para cada mes
    ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Leyenda
    ax.legend()
    return fig

st.title("Análisis de Ventas por producto")

# Carga del archivo CSV
st.sidebar.header("Carga de archivo")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado:
    datos = cargar_datos(archivo_cargado)
    if datos is not None:
        sucursales = ["Todas"] + datos["Sucursal"].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            datos = datos[datos["Sucursal"] == sucursal_seleccionada]

        st.header(f"Análisis de Sucursal: {sucursal_seleccionada}")
        
        # Iterar por cada producto
        productos = datos["Producto"].unique()
        for producto in productos:
            with st.container(border=True):
                prod_datos = datos[datos["Producto"] == producto]

                # Cálculos principales
                prod_datos["Fecha"] = pd.to_datetime(prod_datos["Año"].astype(str) + '-' + prod_datos["Mes"].astype(str) + '-01', errors="coerce")
                ventas_mensuales = prod_datos.groupby("Fecha").sum(numeric_only=True).reset_index()
                precio_anual = (
                    prod_datos.groupby("Año").apply(
                    lambda x: x["Ingreso_total"].sum() / x["Unidades_vendidas"].sum()
                    )
                )
                precio_promedio = round(precio_anual.mean(), 2)

                margen_promedio = round((prod_datos["Ingreso_total"].sum() - prod_datos["Costo_total"].sum()) / prod_datos["Ingreso_total"].sum() * 100, 2)
                unidades_totales = prod_datos["Unidades_vendidas"].sum()

                # Calcular variaciones reales
                variaciones = calcular_variaciones(prod_datos, producto)

                # Mostrar métricas y gráfico dentro de un único contenedor
                st.subheader(producto)
                col1, col2 = st.columns([0.25, 0.75])
                with col1:
                    st.metric("Precio Promedio", f"${precio_promedio:.0f}", f"{variaciones['precio']:.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:.0f}%", f"{variaciones['margen']:.2f}%", delta_color="normal")
                    st.metric("Unidades Vendidas", f"{unidades_totales:,.0f}", f"{variaciones['unidades']:.2f}%")
                with col2:
                    if not ventas_mensuales.empty:
                        fig = generar_grafico(ventas_mensuales, producto)
                        st.pyplot(fig)
                    else:
                        st.warning(f"No hay suficientes datos para generar un gráfico de ventas para {producto}.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
