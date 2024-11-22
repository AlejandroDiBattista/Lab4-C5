import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = https://parcialexcersise-akecznnvhndeuryzrdjbc2.streamlit.app/


def mostrar_informacion_alumno():
    st.sidebar.header("Información")
    st.sidebar.write("**Legajo:** 58.686")
    st.sidebar.write("**Nombre:** Gustavo Rodriguez")
    st.sidebar.write("**Comisión:** C5")


def cargar_datos():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV de datos", type="csv")
    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.sidebar.success("Archivo cargado correctamente.")
        
        if "Año" in datos.columns and "Mes" in datos.columns:
            datos['Fecha'] = pd.to_datetime(datos['Año'].astype(str) + '-' + datos['Mes'].astype(str) + '-01', errors="coerce")
        else:
            st.error("El archivo necesita las columnas 'Año' y 'Mes'.")
            return None
        
        for col in ["Unidades_vendidas", "Ingreso_total", "Costo_total"]:
            if col in datos.columns:
                datos[col] = pd.to_numeric(datos[col], errors="coerce")
        
        return datos
    else:
        st.warning("Por favor, sube un archivo CSV desde la barra lateral.")
        return None

def calcular_metricas(datos, sucursal=None):
    if sucursal:
        datos = datos[datos["Sucursal"] == sucursal]
    
    metricas = datos.groupby("Producto").agg({
        "Unidades_vendidas": "sum",
        "Ingreso_total": "sum",
        "Costo_total": "sum"
    }).reset_index()
    
    metricas["Precio_promedio"] = metricas["Ingreso_total"] / metricas["Unidades_vendidas"]
    metricas["Margen_promedio"] = (metricas["Ingreso_total"] - metricas["Costo_total"]) / metricas["Ingreso_total"]
    
    return metricas

def graficar_ventas(datos, producto):
    datos_producto = datos[datos["Producto"] == producto].groupby("Fecha").agg({
        "Unidades_vendidas": "sum"
    }).reset_index()

    if datos_producto.empty:
        st.warning(f"No hay ventas para el producto: {producto}.")
        return

    if len(datos_producto) > 1:
        datos_producto["Tendencia"] = np.poly1d(np.polyfit(
            np.arange(len(datos_producto)), datos_producto["Unidades_vendidas"], 1
        ))(np.arange(len(datos_producto)))
    
        plt.figure(figsize=(8, 5))
        plt.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=producto, color="blue")
        plt.plot(datos_producto["Fecha"], datos_producto["Tendencia"], linestyle="--", color="red", label="Tendencia")
        plt.xlabel("Fecha")
        plt.ylabel("Unidades Vendidas")
        plt.title(f"Ventas Mensuales - {producto}")
        plt.legend()
        plt.grid()
        st.pyplot(plt)


def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Ventas")
   
    mostrar_informacion_alumno()
    
    datos = cargar_datos()
    if datos is not None:
        sucursal = st.sidebar.selectbox("Seleccionar Sucursal", options=["Todas"] + datos["Sucursal"].unique().tolist())
        datos_filtrados = datos if sucursal == "Todas" else datos[datos["Sucursal"] == sucursal]

        st.header(f"Datos de la Sucursal: {sucursal}")
        metricas = calcular_metricas(datos_filtrados)
        
        for _, fila in metricas.iterrows():
            st.subheader(fila["Producto"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Precio Promedio", f"${fila['Precio_promedio']:.2f}")
            col2.metric("Margen Promedio", f"{fila['Margen_promedio'] * 100:.2f}%")
            col3.metric("Unidades Vendidas", f"{fila['Unidades_vendidas']:.0f}")
            graficar_ventas(datos_filtrados, fila["Producto"])

if __name__ == "__main__":
    main()
