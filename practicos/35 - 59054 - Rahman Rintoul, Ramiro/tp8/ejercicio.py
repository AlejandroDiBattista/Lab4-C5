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

mostrar_informacion_alumno()


st.sidebar.header("Seleccione un archivo CSV")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV", type=["csv"])


    

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.sidebar.subheader("Seleccionar Sucursales")
    sucursales = st.sidebar.selectbox("Seleccione una Sucursal", df['Sucursal'].unique())
    df_sucursal = df[df['Sucursal'] == sucursales]

    st.title(f"Datos de la {sucursales}")
    st.write(df_sucursal)

    resultados = df.groupby("Producto").agg(
            Precio_Promedio=("Ingreso_total", lambda x: np.sum(x) / np.sum(df.loc[x.index, "Unidades_vendidas"])),
            Margen_Promedio=("Ingreso_total", lambda x: np.mean((x - df.loc[x.index, "Costo_total"]) / x) * 100),
            Unidades_Vendidas=("Unidades_vendidas", "sum"),
    ).reset_index()

    st.write("Resultados por Producto:")
    st.dataframe(resultados)
    
    productos = resultados["Producto"].unique()
    for producto in productos:
        
        producto_data = df[df["Producto"] == producto]

        # Ordenar por Año y Mes
        producto_data = producto_data.sort_values(by=["Año", "Mes"])

        eje_temporal = producto_data["Año"].astype(str) + "-" + producto_data["Mes"].astype(str).str.zfill(2)

        # gráfico
        fig, ax = plt.subplots()
        ax.plot(
            eje_temporal,
            producto_data["Unidades_vendidas"],
            label="Unidades Vendidas",
            color="blue",
        )
        ax.set_xticks(eje_temporal[::max(len(eje_temporal) // 10, 1)])  # Reducir etiquetas si hay muchas
        ax.set_xticklabels(eje_temporal[::max(len(eje_temporal) // 10, 1)], rotation=45)
        ax.set_xlabel("Período (Año-Mes)")
        ax.set_ylabel("Unidades Vendidas")

        # Línea de tendencia
        tendencia = np.polyfit(np.arange(len(producto_data)), producto_data["Unidades_vendidas"], 1)
        ax.plot(
            eje_temporal,
            np.polyval(tendencia, np.arange(len(producto_data))),
            label="Tendencia",
            color="red",
        )
        ax.grid(True, linestyle=":", alpha=0.7)

        fig.text(0.02, 0.95, producto, fontsize=18, fontweight="bold", va="top", ha="left")
        
        ax.legend()
        st.pyplot(fig)

