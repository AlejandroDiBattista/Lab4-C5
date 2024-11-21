import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58729.streamlit.app/'

# Función para crear el gráfico de evolución de ventas
def crear_grafico_ventas(datos_producto, producto):
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(ventas_mensuales)), ventas_mensuales['Unidades_vendidas'], label=producto)
    
    # Línea de tendencia
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas']
    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x, tendencia(x), '--', color='red', label='Tendencia')
    
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.legend()
    ax.grid()
    return fig

# Mostrar información del alumno
def mostrar_info_alumno():
    st.sidebar.markdown("**Alumno:** Mathias Lovey")
    st.sidebar.markdown("**Legajo:** 58729")
    st.sidebar.markdown("**Comisión:** C5")

# Cargar archivo CSV
st.sidebar.header("Cargar archivo de datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo:
    datos = pd.read_csv(archivo)
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    productos = datos['Producto'].unique()
    for producto in productos:
        # División clara entre productos
        st.divider()
        st.subheader(producto)
        
        datos_producto = datos[datos['Producto'] == producto]
        
        # Calcular métricas
        precio_promedio = (datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']).mean()
        margen_promedio = ((datos_producto['Ingreso_total'] - datos_producto['Costo_total']) / datos_producto['Ingreso_total']).mean() * 100
        total_unidades = datos_producto['Unidades_vendidas'].sum()
        
        # Variaciones
        precio_promedio_anual = datos_producto.groupby('Año')['Ingreso_total'].sum() / datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_precio = precio_promedio_anual.pct_change().mean() * 100
        
        margen_anual = ((datos_producto.groupby('Año')['Ingreso_total'].sum() - datos_producto.groupby('Año')['Costo_total'].sum()) / datos_producto.groupby('Año')['Ingreso_total'].sum()) * 100
        variacion_margen = margen_anual.pct_change().mean() * 100
        
        unidades_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        variacion_unidades = unidades_anual.pct_change().mean() * 100
        
        # Mostrar métricas
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{variacion_precio:.2f}%")
            st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{variacion_margen:.2f}%")
            st.metric("Unidades Vendidas", f"{total_unidades:,}", f"{variacion_unidades:.2f}%")
        with col2:
            grafico = crear_grafico_ventas(datos_producto, producto)
            st.pyplot(grafico)
else:
    st.write("Por favor, sube un archivo CSV para comenzar.")
    mostrar_info_alumno()