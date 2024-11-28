import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
# url = 'https://parcial2-58894.streamlit.app/'
st.set_page_config(page_title="Análisis de Ventas", layout="wide", initial_sidebar_state="expanded")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.subheader("Información del Alumno")
        st.markdown("**Legajo:** 58.894")
        st.markdown("**Nombre:** Flavia González Nacusse")
        st.markdown("**Comisión:** C5")

def graficar_ventas(datos_producto, producto):
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    ventas_mensuales['Fecha'] = pd.to_datetime(ventas_mensuales['Año'].astype(str) + '-' + ventas_mensuales['Mes'].astype(str))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], label=producto, markersize=5, linestyle='-', color='blue')
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas']
    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(ventas_mensuales['Fecha'], tendencia(x), '--', color='red', label='Tendencia')
    ax.xaxis.set_major_locator(YearLocator())  
    ax.xaxis.set_minor_locator(MonthLocator())  
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.7)  
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)  
    ax.set_title('Evolución de Ventas Mensuales')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.legend()

    return fig

# INTERFAZ PRINCIPAL
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado:
    datos = pd.read_csv(archivo_cargado)
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")

    productos = datos['Producto'].unique()
    for producto in productos:
        
        datos_producto = datos[datos['Producto'] == producto] 
        precio_promedio = (datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']).mean()
        margen_promedio = ((datos_producto['Ingreso_total'] - datos_producto['Costo_total']) / datos_producto['Ingreso_total']).mean() * 100
        total_unidades = datos_producto['Unidades_vendidas'].sum()

        precio_promedio_anual = datos_producto.groupby('Año')['Ingreso_total'].sum() / datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_precio = precio_promedio_anual.pct_change().mean() * 100
        
        margen_anual = ((datos_producto.groupby('Año')['Ingreso_total'].sum() - datos_producto.groupby('Año')['Costo_total'].sum()) / datos_producto.groupby('Año')['Ingreso_total'].sum()) * 100
        variacion_margen = margen_anual.pct_change().mean() * 100
        
        unidades_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_unidades = unidades_anual.pct_change().mean() * 100

        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"## **{producto}**") 
                st.metric(
                    "Precio Promedio",
                    f"${precio_promedio:,.0f}".replace(",", "."), 
                    f"{variacion_precio:.2f}%".replace(",", ".")
                )
                st.metric(
                    "Margen Promedio",
                    f"{margen_promedio:.0f}%".replace(",", "."),
                    f"{variacion_margen:.2f}%".replace(",", ".")
                )
                st.metric(
                    "Unidades Vendidas",
                    f"{int(total_unidades):,}".replace(",", "."), 
                    f"{variacion_unidades:.2f}%".replace(",", ".")
                )
            with col2:
                grafico = graficar_ventas(datos_producto, producto)
                st.pyplot(grafico)
else:
    st.info("Por favor, sube un archivo CSV con los datos de ventas. Desde la barra lateral.")
    mostrar_informacion_alumno()
