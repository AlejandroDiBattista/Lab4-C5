import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://trabajo-practico-8.streamlit.app/'

def info_estudiante():
    with st.container():
        st.markdown('*Legajo:* 59.070')
        st.markdown('*Nombre:* Maia Agostina Ladina')
        st.markdown('*Comisión:* C5')
info_estudiante()

def graficar_ventas(datos, producto):

    ventas_mensuales = datos.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas']

    ax.plot(x, y, label=f'Ventas de {producto}', color='blue')

    coef = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x, tendencia(x), linestyle='--', color='red', label='Tendencia')

    ax.set_title('Tendencia de Ventas Mensuales', fontsize=16)
    ax.set_xlabel('Meses', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)
    ax.set_xticks(x)
    etiquetas = [f"{fila.Año}" if fila.Mes == 1 else "" for fila in ventas_mensuales.itertuples()]
    ax.set_xticklabels(etiquetas, rotation=45)
    ax.legend()
    ax.grid(alpha=0.5)

    return fig

st.sidebar.header("Cargar archivo de datos")
archivo_csv = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if archivo_csv:
    datos = pd.read_csv(archivo_csv)

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal = st.sidebar.selectbox("Sucursal", sucursales)

    if sucursal != "Todas":
        datos = datos[datos['Sucursal'] == sucursal]
        st.title(f"Datos de la Sucursal: {sucursal}")
    else:
        st.title("Datos de Todas las Sucursales")

    for producto in datos['Producto'].unique():
        with st.container():
            st.subheader(producto)
            datos_producto = datos[datos['Producto'] == producto]

            if 'Ganancia' not in datos_producto.columns or 'Margen' not in datos_producto.columns:
                datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
                datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100

            datos_producto['PrecioProm'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_medio = datos_producto['PrecioProm'].mean()
            margen_medio = datos_producto['Margen'].mean()
            total_unidades = datos_producto['Unidades_vendidas'].sum()

            if not datos_producto.empty and 'Año' in datos_producto.columns:
                cambio_precio = datos_producto.groupby('Año')['PrecioProm'].mean().pct_change().mean() * 100
                cambio_margen = datos_producto.groupby('Año')['Margen'].mean().pct_change().mean() * 100
                cambio_unidades = datos_producto.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100
            else:
                cambio_precio = cambio_margen = cambio_unidades = 0

            col1, col2 = st.columns([0.4, 0.6])

            with col1:
                st.metric("Precio Promedio", f"${precio_medio:,.2f}", f"{cambio_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_medio:.2f}%", f"{cambio_margen:.2f}%")
                st.metric("Total Unidades", f"{total_unidades:,.0f}", f"{cambio_unidades:.2f}%")

            with col2:
                grafico = graficar_ventas(datos_producto, producto)
                st.pyplot(grafico)
else:
    st.subheader("Carga un archivo CSV en la barra lateral para empezar.")