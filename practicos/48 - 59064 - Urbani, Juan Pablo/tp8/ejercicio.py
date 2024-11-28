import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCIÓN: Dirección de publicación de la aplicación url = https://urbanijuanpablo.streamlit.app/ ##

def mostrar_informacion_alumno():
    st.header("Por favor, sube un archivo CSV para comenzar.")
    with st.container():
        st.markdown('**Legajo:** 59064')
        st.markdown('**Nombre:** Juan Pablo Urbani')
        st.markdown('**Comisión:** C5')

st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if not uploaded_file:
    mostrar_informacion_alumno()
else:
    try:
        df = pd.read_csv(uploaded_file)
        columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
        if not all(col in df.columns for col in columnas_requeridas):
            st.error("El archivo CSV no contiene las columnas requeridas.")
        else:
            df["Precio_promedio"] = df["Ingreso_total"] / df["Unidades_vendidas"]
            df["Margen_promedio"] = (df["Ingreso_total"] - df["Costo_total"]) / df["Ingreso_total"]
            df["Mes_anio"] = df["Año"].astype(str) + "-" + df["Mes"].astype(str).str.zfill(2)
            df["Cambio_precio"] = df.groupby("Producto")["Precio_promedio"].pct_change() * 100
            df["Cambio_margen"] = df.groupby("Producto")["Margen_promedio"].pct_change() * 100
            df["Cambio_unidades"] = df.groupby("Producto")["Unidades_vendidas"].pct_change() * 100
            df.fillna({"Cambio_precio": 0, "Cambio_margen": 0, "Cambio_unidades": 0}, inplace=True)

            sucursales = ["Todas"] + sorted(df["Sucursal"].unique())
            sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

            if sucursal_seleccionada == "Todas":
                st.header("Datos de Todas las sucursales")
            else:
                st.header(f"Datos de {sucursal_seleccionada}")

            if sucursal_seleccionada != "Todas":
                df = df[df["Sucursal"] == sucursal_seleccionada]

            for producto in df["Producto"].unique():
                df_producto = df[df["Producto"] == producto]
                precio_promedio = df_producto["Precio_promedio"].mean()
                margen_promedio = df_producto["Margen_promedio"].mean()
                unidades_vendidas = df_producto["Unidades_vendidas"].sum()
                cambio_precio = df_producto["Cambio_precio"].iloc[-1]
                cambio_margen = df_producto["Cambio_margen"].iloc[-1]
                cambio_unidades = df_producto["Cambio_unidades"].iloc[-1]

                with st.container(border=True):
                    st.subheader(f"{producto}")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Precio Promedio", f"${precio_promedio:.2f}", f"{cambio_precio:.2f}%")
                        st.metric("Margen Promedio", f"{margen_promedio:.2%}", f"{cambio_margen:.2f}%")
                        st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}", f"{cambio_unidades:.2f}%")
                    with col2:
                        df_producto["Año-Mes"] = pd.to_datetime(df_producto["Año"].astype(str) + "-" + df_producto["Mes"].astype(str).str.zfill(2))
                        df_producto["Unidades_suavizadas"] = df_producto["Unidades_vendidas"].rolling(window=3, center=True).mean()
                        df_producto["Fecha_Num"] = (df_producto["Año-Mes"] - df_producto["Año-Mes"].min()).dt.days
                        x = df_producto["Fecha_Num"].values
                        y = df_producto["Unidades_vendidas"].values
                        m = (np.sum(x * y) - len(x) * np.mean(x) * np.mean(y)) / (np.sum(x**2) - len(x) * np.mean(x)**2)
                        b = np.mean(y) - m * np.mean(x)
                        df_producto["Tendencia"] = m * x + b

                        fig, ax = plt.subplots(figsize=(6, 4)) 
                        ax.plot(
                            df_producto["Año-Mes"],
                            df_producto["Unidades_suavizadas"],
                            label="Unidades Suavizadas",
                            color="blue",
                            linewidth=2
                        )
                        ax.plot(
                            df_producto["Año-Mes"],
                            df_producto["Tendencia"],
                            label="Tendencia",
                            color="red",
                            linestyle="--",
                            linewidth=2
                        )
                        ax.set_title(f"Evolución de Ventas - {producto}", fontsize=12)
                        ax.set_xlabel("Año-Mes", fontsize=10)
                        ax.set_ylabel("Unidades Vendidas", fontsize=10)
                        ax.grid(visible=True, which="both", linestyle="--", alpha=0.6)
                        ax.legend(fontsize=8)
                        plt.xticks(rotation=45, ha="right")

                        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
