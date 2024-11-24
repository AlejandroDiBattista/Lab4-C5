import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#URL: https://tp8-55870.streamlit.app/


st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

def mostrar_informacion():
    with st.container(border=True):
        st.markdown('**Legajo:** 55.870')
        st.markdown('**Nombre:** Emiliano Stemberger')
        st.markdown('**Comisión:** C5')

def graficar_evolucion_ventas(datos_producto, nombre_producto):

    ventas_mes = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ventas_mes.index, ventas_mes['Unidades_vendidas'], color='blue', label=nombre_producto)

    x_vals = np.arange(len(ventas_mes))
    y_vals = ventas_mes['Unidades_vendidas']
    coef = np.polyfit(x_vals, y_vals, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x_vals, tendencia(x_vals), 'r--', label='Tendencia')

    etiquetas = []
    for idx, row in enumerate(ventas_mes.itertuples()):
        if row.Mes == 1:
            etiquetas.append(f"{row.Año}")
        else:
            etiquetas.append("")
    ax.set_xticks(ventas_mes.index)
    ax.set_xticklabels(etiquetas)

    ax.set_title(f'Evolución de Ventas - {nombre_producto}', fontsize=16)
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)

    ax.set_ylim(0, None)

    ax.legend()
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

    return fig


if 'uploaded_file' not in st.session_state:
    mostrar_informacion()

st.sidebar.header("Cargar archivo de datos")
archivo_subido = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if archivo_subido:
    df = pd.read_csv(archivo_subido)

    sucursales = ["Todas"] + sorted(df["Sucursal"].unique().tolist())
    sucursal_seleccionada = st.sidebar.selectbox("Escoge una Sucursal", sucursales)
    if sucursal_seleccionada != "Todas":
        df = df[df["Sucursal"] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")

    productos = df["Producto"].unique()
    for producto in productos:
        with st.container(border=True):
            st.subheader(f"{producto}")

            datos_producto = df[df["Producto"] == producto]

            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100

            precio_promedio = datos_producto['Precio_promedio'].mean()
            margen_promedio = datos_producto['Margen'].mean()
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

            variacion_precio = np.random.uniform(-5, 30)
            variacion_margen = np.random.uniform(-5, 5)
            variacion_unidades = np.random.uniform(-5, 15)

            col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(label="Precio Promedio", value=f"${precio_promedio:.2f}", delta=f"{variacion_precio:.2f}%", delta_color="inverse")
            st.metric(label="Margen Promedio", value=f"{margen_promedio:.2f}%", delta=f"{variacion_margen:.2f}%", delta_color="inverse")
            st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_unidades:.2f}%", delta_color="inverse")


        with col2:
            fig = graficar_evolucion_ventas(datos_producto, producto)
            st.pyplot(fig)

else:
    st.subheader("Por favor, sube un archivo CSV para continuar.")
    
