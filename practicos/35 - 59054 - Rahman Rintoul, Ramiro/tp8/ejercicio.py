import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
url = 'https://tp8-lab-c5-3hksyedewsvqjsvscna2pt.streamlit.app'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.054')
        st.markdown('**Nombre:** Ramiro Rahman Rintoul')
        st.markdown('**Comisión:** C5')



st.sidebar.header("Seleccione un archivo CSV")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV", type=["csv"])

if uploaded_file is None:
    mostrar_informacion_alumno()


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
    if not set(columnas_requeridas).issubset(df.columns):
        st.error(f"El archivo debe contener las columnas: {', '.join(columnas_requeridas)}")
        st.stop()

    df["Año-Mes"] = df["Año"].astype(str) + "-" + df["Mes"].astype(str).str.zfill(2)

    sucursales = st.sidebar.selectbox("Seleccione una Sucursal", ["Todas las sucursales"] + sorted(df["Sucursal"].unique()))
    if sucursales != "Todas las sucursales":
        df = df[df["Sucursal"] == sucursales]

    st.title(f"Datos de Ventas - {sucursales}")

    for producto in df["Producto"].unique():
        producto_data = df[df["Producto"] == producto]

        producto_data = producto_data.sort_values(by=["Año", "Mes"])
        producto_data["Precio Promedio"] = producto_data["Ingreso_total"] / producto_data["Unidades_vendidas"]
        producto_data["Margen Promedio"] = (producto_data["Ingreso_total"] - producto_data["Costo_total"]) / producto_data["Ingreso_total"]
        
        precio_promedio = producto_data["Precio Promedio"].mean()
        margen_promedio = producto_data["Margen Promedio"].mean()
        unidades_vendidas = producto_data["Unidades_vendidas"].sum()

        producto_data["Cambio Precio (%)"] = producto_data["Precio Promedio"].pct_change() * 100
        producto_data["Cambio Margen (%)"] = producto_data["Margen Promedio"].pct_change() * 100
        producto_data["Cambio Unidades (%)"] = producto_data["Unidades_vendidas"].pct_change() * 100
        
        cambio_precio = producto_data["Cambio Precio (%)"].iloc[-1]
        cambio_margen = producto_data["Cambio Margen (%)"].iloc[-1]
        cambio_unidades = producto_data["Cambio Unidades (%)"].iloc[-1]

        with st.container(border=True):
            st.subheader(producto)

            col1, col2 = st.columns([1, 3])
            with col1:
                col1.metric(
                    "Precio Promedio", 
                    f"${precio_promedio:,.2f}", 
                    f"{producto_data['Cambio Precio (%)'].iloc[-1]:.2f}%"
                )
                col1.metric(
                    "Margen Promedio", 
                    f"{margen_promedio * 100:.2f}%", 
                    f"{producto_data['Cambio Margen (%)'].iloc[-1]:.2f}%"
                )
                col1.metric(
                    "Unidades Vendidas", 
                    f"{unidades_vendidas:,}", 
                    f"{producto_data['Cambio Unidades (%)'].iloc[-1]:.2f}%"
                )

            with col2:      
                eje_temporal = producto_data["Año-Mes"]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(eje_temporal, producto_data["Unidades_vendidas"], label="Unidades Vendidas", color="blue")
        
                z = np.polyfit(range(len(producto_data)), producto_data["Unidades_vendidas"], 1)
                tendencia = np.poly1d(z)(range(len(producto_data)))
                ax.plot(eje_temporal, tendencia, label="Tendencia", linestyle="--", color="red")
        
                ax.set_title(f"Evolución de Ventas - {producto}")
                ax.set_xlabel("Año-Mes")
                ax.set_ylabel("Unidades Vendidas")
                ax.set_xticks(eje_temporal[::max(len(eje_temporal) // 10, 1)])
                ax.set_xticklabels(eje_temporal[::max(len(eje_temporal) // 10, 1)], rotation=45, ha="right")
                ax.legend()
                ax.grid(True)
        
                st.pyplot(fig)



    