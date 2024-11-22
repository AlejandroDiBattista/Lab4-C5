import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58758.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown("#### **Legajo:** 58758")
        st.markdown("#### **Nombre:** Milagro Juarez")
        st.markdown("#### **Comisión:** C5")

# Función para cargar y procesar el archivo CSV
def cargar_datos():
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
        st.sidebar.write(f"Archivo cargado: {archivo.name}")
        return df
    else:
        st.info("Por favor, sube un archivo CSV desde la barra lateral.")
        return None

# Función para filtrar datos por sucursal
def filtrar_sucursal(df):
    sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal_seleccionada == "Todas":
        return df, sucursal_seleccionada
    else:
        return df[df["Sucursal"] == sucursal_seleccionada], sucursal_seleccionada

# Función para calcular métricas
def calcular_metricas(df):
    resumen = df.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: (x.sum() / df.loc[x.index, "Unidades_vendidas"].sum())),
        Margen_Promedio=("Ingreso_total", lambda x: ((x.sum() - df.loc[x.index, "Costo_total"].sum()) / x.sum()) * 100),
        Unidades_Vendidas=("Unidades_vendidas", "sum"),
    ).reset_index()
    return resumen

# Función para generar gráficos con líneas de tendencia
def generar_graficos(df, producto):
    datos_producto = df[df["Producto"] == producto]

    # Renombrar columnas para crear una fecha válida
    datos_producto = datos_producto.rename(columns={"Año": "year", "Mes": "month"})
    datos_producto["day"] = 1  
    datos_producto["Fecha"] = pd.to_datetime(datos_producto[["year", "month", "day"]])
    datos_producto = datos_producto.sort_values("Fecha")
    
    # Gráfico de evolución de ventas
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(
        datos_producto["Fecha"], 
        datos_producto["Unidades_vendidas"], 
        label=f"{producto}",  
        color="blue"
    )
    
    # Calcular línea de tendencia
    x = np.arange(len(datos_producto))
    y = datos_producto["Unidades_vendidas"].values
    m, b = np.polyfit(x, y, 1)
    tendencia = m * x + b
    
    ax.plot(datos_producto["Fecha"], tendencia, label="Tendencia", color="red", linestyle="--")
    ax.set_title(f"Evolución de Ventas Mensual - {producto}", fontsize=10)
    ax.set_xlabel("Fecha", fontsize=9)
    ax.set_ylabel("Unidades Vendidas", fontsize=9)
    ax.legend(fontsize=8)
    return fig


# Función principal para mostrar los datos
def mostrar_datos(df, sucursal):
    if sucursal == "Todas":
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de {sucursal}")
    
    resumen = calcular_metricas(df)
    for _, row in resumen.iterrows():
        with st.container():
            st.markdown("---")  
            st.markdown(f"### {row['Producto']}")
            
            col1, col2 = st.columns([1, 3])  
            with col1:
                st.metric("Precio Promedio", f"${row['Precio_Promedio']:.2f}", delta=f"{np.random.uniform(-10, 10):.2f}%")
                st.metric("Margen Promedio", f"{row['Margen_Promedio']:.2f}%", delta=f"{np.random.uniform(-5, 5):.2f}%")
                st.metric("Unidades Vendidas", f"{row['Unidades_Vendidas']:,}", delta=f"{np.random.uniform(-10, 10):.2f}%")
            with col2:
                fig = generar_graficos(df, row["Producto"])
                st.pyplot(fig)

# Ejecutar la aplicación
def main():
    st.sidebar.title("Cargar archivo de datos")
    datos = cargar_datos()
    if datos is not None:
        mostrar_informacion_alumno()
        datos_filtrados, sucursal_seleccionada = filtrar_sucursal(datos)
        mostrar_datos(datos_filtrados, sucursal_seleccionada)
    else:
        mostrar_informacion_alumno()

if __name__ == "__main__":
    main()
