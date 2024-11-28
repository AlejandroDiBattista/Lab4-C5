import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


#url = https://tp8---juncos-gonzalo-parcial.streamlit.app/ 

# Información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('Legajo: 58.721')
        st.markdown('Nombre: Juncos Gonzalo')
        st.markdown('Comisión: C5')

def calcular_metricas(df):
    df['Ingreso_total'] = pd.to_numeric(df['Ingreso_total'], errors='coerce')
    df['Unidades_vendidas'] = pd.to_numeric(df['Unidades_vendidas'], errors='coerce')
    df['Costo_total'] = pd.to_numeric(df['Costo_total'], errors='coerce')

    df = df[(df['Unidades_vendidas'] > 0) & (df['Ingreso_total'] > 0)]
    
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen'] = ((df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']) * 100
    return df

def calcular_cambios_anuales(df):
    return {
        producto: {
            "variacion_precio_promedio": datos.groupby('Año')['Precio_promedio'].mean().pct_change().mean() * 100,
            "variacion_margen_promedio": datos.groupby('Año')['Margen'].mean().pct_change().mean() * 100,
            "variacion_unidades_vendidas": datos.groupby('Año')['Unidades_vendidas'].sum().pct_change().mean() * 100
        }
        for producto, datos in df.groupby('Producto')
    }


def procesar_datos(df, sucursal_seleccionada):
    columnas_requeridas = ['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
    if not all(col in df.columns for col in columnas_requeridas):
        st.error(f"El archivo no contiene las columnas requeridas: {', '.join(columnas_requeridas)}")
        return

    df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
    df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')
    df = df.dropna(subset=['Año', 'Mes'])

    if sucursal_seleccionada != "Todas":
        df = df[df['Sucursal'] == sucursal_seleccionada]

    df = calcular_metricas(df)
    variaciones = calcular_cambios_anuales(df)

    for producto in df['Producto'].unique():
        datos_producto = df[df['Producto'] == producto]

        ingresos_totales = datos_producto['Ingreso_total'].sum()
        unidades_totales = datos_producto['Unidades_vendidas'].sum()
        costos_totales = datos_producto['Costo_total'].sum()

        ingresos_totales_producto = datos_producto['Ingreso_total'].sum()
        unidades_totales_producto = datos_producto['Unidades_vendidas'].sum()
      
        precio_promedio = round(np.mean(datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']))


        margen_promedio = round(((ingresos_totales - costos_totales) / ingresos_totales) * 100) if ingresos_totales != 0 else 0


        variacion_precio = variaciones[producto]["variacion_precio_promedio"]
        variacion_margen = variaciones[producto]["variacion_margen_promedio"]
        variacion_unidades = variaciones[producto]["variacion_unidades_vendidas"]

        with st.container():
            st.markdown(f"### {producto}")
            col1, col2 = st.columns([1, 2])
            with col1:
                precio_formateado = f"{precio_promedio:,.2f}".replace(",", ".")
                unidades_formateadas = f"{int(unidades_totales):,}".replace(",", ".")

                st.metric("Precio Promedio", f"$ {precio_formateado}", delta=f"{variacion_precio:+.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.2f}%", delta=f"{variacion_margen:+.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_formateadas}", delta=f"{variacion_unidades:+.2f}%")
            with col2:
                datos_producto['Fecha'] = pd.to_datetime({'year': datos_producto['Año'], 'month': datos_producto['Mes'], 'day': 1})
                datos_producto = datos_producto.sort_values('Fecha')
                if not datos_producto.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], label="Unidades Vendidas", color="blue")
                    
                    fechas = (datos_producto['Fecha'] - datos_producto['Fecha'].min()).dt.days
                    z = np.polyfit(fechas, datos_producto['Unidades_vendidas'], 1)
                    p = np.poly1d(z)
                    ax.plot(datos_producto['Fecha'], p(fechas), label="Tendencia", color="red", linestyle="--")
                    
                    ax.set_title(f"Evolución de Ventas - {producto}")
                    ax.set_xlabel("Fecha")
                    ax.set_ylabel("Unidades Vendidas")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

st.set_page_config(layout="wide")
st.title("Datos de Todas las Sucursales")
st.sidebar.title("Cargar archivo de datos")

archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
if archivo_cargado:
    datos = pd.read_csv(archivo_cargado)
    sucursales = ["Todas"] + sorted(datos['Sucursal'].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    procesar_datos(datos, sucursal_seleccionada)
else:
    st.sidebar.warning("Sube un archivo CSV para comenzar.")

mostrar_informacion_alumno()













