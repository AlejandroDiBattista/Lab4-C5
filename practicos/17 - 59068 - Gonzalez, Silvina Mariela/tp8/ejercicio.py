import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-59068.streamlit.app/'

st.set_page_config(page_title="Evolución Ventas")
st.sidebar.title("Cargar archivos de datos")

archivo = st.sidebar.file_uploader("Subir archivo.csv", type=["csv"])
seleccion_sucursal = st.sidebar.selectbox("Seleccionar sucursal", options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])

if archivo:
    df = pd.read_csv(archivo)
    df["Fecha"] = pd.to_datetime(df["Año"].astype(str) + "-" + df["Mes"].astype(str) + "-01")

    if seleccion_sucursal != "Todas":
        df = df[df["Sucursal"] == seleccion_sucursal]

    st.title(f"Datos de {seleccion_sucursal}")
    st.markdown("""
    <hr style="border: .1rem solid #808080;"/>
    """, unsafe_allow_html=True)

    productos = df["Producto"].unique()
    for producto in productos:
        df_producto = df[df["Producto"] == producto]
        df_producto = df_producto.sort_values("Fecha")

        df_producto["Precio_promedio"] = df_producto["Ingreso_total"] / df_producto["Unidades_vendidas"]
        df_producto["Ganancia"] = df_producto["Ingreso_total"] - df_producto["Costo_total"]
        df_producto["Margen"] = (df_producto["Ganancia"] / df_producto["Ingreso_total"]) * 100

        precio = df_producto["Precio_promedio"].mean()
        margen = df_producto["Margen"].mean()
        unidades_totales = df_producto["Unidades_vendidas"].sum()

        precio_anual = df_producto.groupby("Año")["Precio_promedio"].mean()
        variacion_precio = precio_anual.pct_change().mean() * 100

        margen_anual = df_producto.groupby("Año")["Margen"].mean()
        variacion_margen = margen_anual.pct_change().mean() * 100

        unidades_por_año = df_producto.groupby("Año")["Unidades_vendidas"].sum()
        variacion_unidades = unidades_por_año.pct_change().mean() * 100

        col1, col2 = st.columns([2, 6])

        with col1:
            st.subheader(f"{producto}")
            st.metric("Precio Promedio", f"${round(precio, 2):,}", f"{variacion_precio:+.2f}%")
            st.metric("Margen Promedio", f"{round(margen, 2)}%", f"{variacion_margen:+.2f}%")
            st.metric("Unidades Vendidas", f"{int(unidades_totales):,}", f"{variacion_unidades:+.2f}%")

        with col2:
            ventas_por_fecha = df_producto.groupby(["Año", "Mes"])["Unidades_vendidas"].sum().reset_index()
            ventas_por_fecha["Fecha"] = pd.to_datetime(ventas_por_fecha["Año"].astype(str) + "-" + ventas_por_fecha["Mes"].astype(str) + "-01")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(ventas_por_fecha["Fecha"], ventas_por_fecha["Unidades_vendidas"], label="Unidades Vendidas", color="blue")

            x = np.arange(len(ventas_por_fecha))
            y = ventas_por_fecha["Unidades_vendidas"]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(ventas_por_fecha["Fecha"], p(x), linestyle="--", color="red", label="Tendencia")

            etiquetas = []
            ultimo_año = None
            for fecha in ventas_por_fecha["Fecha"]:
                año = fecha.year
                if año != ultimo_año:
                    etiquetas.append(str(año))
                    ultimo_año = año
                else:
                    etiquetas.append("") 

            ax.set_xticks(ventas_por_fecha["Fecha"])
            ax.set_xticklabels(etiquetas, rotation=45, ha="right")
            ax.set_title(f"Evolución de Ventas Mensual")
            ax.set_xlabel("Año-Mes")
            ax.legend(title='Producto')
            ax.set_ylabel("Unidades Vendidas")
            ax.set_ylim(0, None)
            ax.grid(True)

            st.pyplot(fig)

        st.markdown("""
        <hr style="border: .1rem solid #808080;"/>
        """, unsafe_allow_html=True)

else:
    st.info("Debes subir un archivo con extensión CSV para cargar los datos.")
    def mostrar_informacion_alumno():
        with st.container():
            st.markdown('**Legajo:** 59.068')
            st.markdown('**Nombre:** Silvina Mariela González')
            st.markdown('**Comisión:** C5')

    mostrar_informacion_alumno()
