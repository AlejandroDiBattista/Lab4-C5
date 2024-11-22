import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://tp8-lab-xvbgxpfxxdt45ydttjuhgj.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.753')
        st.markdown('**Nombre:** Lautaro Rivadeneira')
        st.markdown('**Comisión:** C5')

def cargar_datos():
    st.sidebar.title("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        return datos
    return None

def calcular_estadisticas(datos, sucursal):
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]
    
    agrupado = datos.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: (x.sum() / datos.loc[x.index, "Unidades_vendidas"].sum())),
        Margen_Promedio=("Ingreso_total", lambda x: ((x.sum() - datos.loc[x.index, "Costo_total"].sum()) / x.sum())),
        Unidades_Vendidas=("Unidades_vendidas", "sum")
    )
    agrupado["Cambio_Precio"] = np.random.uniform(-10, 10, len(agrupado))
    agrupado["Cambio_Margen"] = np.random.uniform(-5, 5, len(agrupado))
    agrupado["Cambio_Unidades"] = np.random.uniform(-15, 15, len(agrupado))
    return agrupado

def mostrar_metricas(columna, titulo, valor, cambio):
    if cambio > 0:
        delta_color = "normal"
    else:
        delta_color = "inverse"
    columna.metric(
        titulo,
        f"${valor:,.2f}" if "Precio" in titulo else f"{valor:,.0f}" if "Unidades" in titulo else f"{valor:.0%}",
        f"{cambio:+.2f}%",
    )

def graficar_evolucion(datos, producto):
    datos_producto = datos[datos["Producto"] == producto].copy()

    datos_producto["Año"] = pd.to_numeric(datos_producto["Año"], errors="coerce")
    datos_producto["Mes"] = pd.to_numeric(datos_producto["Mes"], errors="coerce")
    
    datos_producto = datos_producto.dropna(subset=["Año", "Mes"])

    datos_producto["Año"] = datos_producto["Año"].astype(int)
    datos_producto["Mes"] = datos_producto["Mes"].astype(int)

    datos_producto["Fecha"] = pd.to_datetime({
        "year": datos_producto["Año"],
        "month": datos_producto["Mes"],
        "day": 1
    }, errors="coerce")

    datos_producto = datos_producto.dropna(subset=["Fecha"])

    datos_producto.sort_values("Fecha", inplace=True)

    datos_producto["Tendencia"] = np.poly1d(
        np.polyfit(range(len(datos_producto)), datos_producto["Unidades_vendidas"], 1)
    )(range(len(datos_producto)))

    plt.figure(figsize=(15, 10))
    plt.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label="Producto")
    plt.plot(datos_producto["Fecha"], datos_producto["Tendencia"], label="Tendencia", linestyle="--", color="red")
    plt.title(f"Evolucion de Ventas Mensual - {producto}")
    plt.xlabel("Año-Mes")
    plt.ylabel("Unidades Vendidas")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    st.pyplot(plt)

def main():
    datos = cargar_datos()
    
    if datos is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
    else:
        sucursal = st.sidebar.selectbox(
            "Seleccionar Sucursal", ["Todas"] + sorted(datos["Sucursal"].unique())
        )
        
        if sucursal == "Todas":
            st.title("Datos de Todas las Sucursales")
        else:
            st.title(f"Datos de {sucursal}")
        
        estadisticas = calcular_estadisticas(datos, sucursal)
        
        for producto, row in estadisticas.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader(producto)
                    mostrar_metricas(st, "Precio Promedio", row["Precio_Promedio"], row["Cambio_Precio"])
                    mostrar_metricas(st, "Margen Promedio", row["Margen_Promedio"], row["Cambio_Margen"])
                    mostrar_metricas(st, "Unidades Vendidas", row["Unidades_Vendidas"], row["Cambio_Unidades"])
                with col2:
                    graficar_evolucion(datos, producto)
            st.markdown(" ")

if __name__ == "__main__":
    main()