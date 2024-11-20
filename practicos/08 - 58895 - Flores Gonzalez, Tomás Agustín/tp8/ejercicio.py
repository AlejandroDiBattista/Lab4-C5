import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58895.streamlit.app/'

def mostrar_informacion_alumno():
    st.sidebar.header("Cargar archivo de datos")
    st.markdown("""
    ### Sube un archivo CSV.
    """)
    with st.container(border=True):
        st.markdown('**Legajo:** 58895')
        st.markdown('**Nombre:** Flores Gonzalez Tomas Agustin')
        st.markdown('**Comisión:** C5')

def cargar_datos():
    uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    return None

def mostrar_datos(data):
    st.sidebar.subheader("Seleccionar Sucursal")
    sucursales = ["Todas"] + sorted(data["Sucursal"].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if sucursal_seleccionada != "Todas":
        data = data[data["Sucursal"] == sucursal_seleccionada]

    st.header("Datos de Todas las Sucursales" if sucursal_seleccionada == "Todas" else f"Datos de {sucursal_seleccionada}")
    
    for producto in data["Producto"].unique():
        producto_data = data[data["Producto"] == producto]

        
        precio_promedio = producto_data["Ingreso_total"].sum() / producto_data["Unidades_vendidas"].sum()
        margen_promedio = (producto_data["Ingreso_total"].sum() - producto_data["Costo_total"].sum()) / producto_data["Ingreso_total"].sum() * 100
        unidades_totales = producto_data["Unidades_vendidas"].sum()

        
        grafico_datos = producto_data.groupby(["Año", "Mes"]).sum().reset_index()
        grafico_datos["Fecha"] = pd.to_datetime(grafico_datos["Año"].astype(str) + "-" + grafico_datos["Mes"].astype(str))
        grafico_datos.sort_values("Fecha", inplace=True)

        unidades_vendidas = grafico_datos["Unidades_vendidas"]
        tendencia = unidades_vendidas.rolling(window=6, min_periods=1).mean()

        
        with st.container():
            st.subheader(producto)
            col1, col2, col3 = st.columns(3)

            col1.metric("Precio Promedio", f"${precio_promedio:,.2f}")
            col2.metric("Margen Promedio", f"{margen_promedio:.2f}%")
            col3.metric("Unidades Vendidas", f"{unidades_totales:,.0f}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(grafico_datos["Fecha"], unidades_vendidas, label="Unidades Vendidas", color="blue", linewidth=2)
            ax.plot(grafico_datos["Fecha"], tendencia, label="Tendencia", color="red", linestyle="--", linewidth=2)
            ax.set_title(f"Evolución de Ventas Mensual - {producto}", fontsize=14)
            ax.set_xlabel("Período (Año-Mes)", fontsize=12)
            ax.set_ylabel("Unidades Vendidas", fontsize=12)
            ax.legend()
            ax.grid(alpha=0.5)
            plt.xticks(rotation=45)
            st.pyplot(fig)


def main():
    st.set_page_config(layout="wide", page_title="Análisis de Ventas")
    mostrar_informacion_alumno()
    data = cargar_datos()

    if data is not None:
        mostrar_datos(data)
    else:
        st.warning("Carga un archivo CSV para mostrar los datos.")

if __name__ == "__main__":
    main()