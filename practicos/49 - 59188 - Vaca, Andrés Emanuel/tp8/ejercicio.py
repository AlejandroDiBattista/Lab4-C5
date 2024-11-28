import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import MonthLocator
from matplotlib.ticker import MultipleLocator


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://qg5wmyibamihzzm6f3smkm.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 59.188')
        st.markdown('**Nombre:** Vaca Andrés')
        st.markdown('**Comisión:** C5')


def cargar_datos():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV ", type=["csv"])

    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
            return datos
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    return None



def filtrar_por_sucursal(datos):
    sucursales = ["Todas"] + list(datos['Sucursal'].unique())
    seleccion = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if seleccion != "Todas":
        datos = datos[datos['Sucursal'] == seleccion]
    return datos, seleccion


def mostrar_graficos(datos, sucursal):
    st.title(f"Análisis de Ventas: {sucursal}")
    datos['Fecha'] = pd.to_datetime(
        datos['Año'].astype(str) + '-' + datos['Mes'].apply(lambda x: str(x).zfill(2))
    )

    productos = datos['Producto'].unique()

    for producto in productos:
        datos_producto = datos[datos['Producto'] == producto]
        col1, col2 = st.columns([1, 2])

        with st.container(border=True):
            col1, col2 = st.columns([1, 3])


        with col1:
             st.subheader(f"Producto: {producto}")

             datos_producto['Precio_unitario'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
             promedio_precio = datos_producto['Precio_unitario'].mean()

             promedio_precio_por_año = datos_producto.groupby('Año')['Precio_unitario'].mean()
             cambio_promedio_precio_anual = promedio_precio_por_año.pct_change().mean() * 100

             datos_producto['Beneficio'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
             datos_producto['Porcentaje_ganancia'] = (datos_producto['Beneficio'] / datos_producto['Ingreso_total']) * 100
             promedio_margen = datos_producto['Porcentaje_ganancia'].mean()

             promedio_margen_por_año = datos_producto.groupby('Año')['Porcentaje_ganancia'].mean()
             cambio_margen_promedio_anual = promedio_margen_por_año.pct_change().mean() * 100

             total_vendido = datos_producto['Unidades_vendidas'].sum()
             unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
             variacion_anual_unidades_vendidas = unidades_por_año.pct_change().mean() * 100
 
           
             st.metric( label="Precio Medio", value=f"${promedio_precio:,.0f}".replace(",", "."),  delta=f"{cambio_promedio_precio_anual:.2f}%")
             
             st.metric( label="Margen Medio",  value=f"{promedio_margen:.0f}%".replace(",", "."), delta=f"{cambio_margen_promedio_anual:.2f}%" )
             
             st.metric(label="Unidades Totales", value=f"{total_vendido:,}".replace(",", "."), delta=f"{variacion_anual_unidades_vendidas:.2f}%")

      
        with col2:
            datos_producto["Fecha_Num"] = (datos_producto["Fecha"] - datos_producto["Fecha"].min()).dt.days
            x = datos_producto["Fecha_Num"].values
            y = datos_producto["Unidades_vendidas"].values
            

            if len(x) > 1: 
                m = (np.sum(x * y) - len(x) * np.mean(x) * np.mean(y)) / (np.sum(x**2) - len(x) * np.mean(x)**2)
                b = np.mean(y) - m * np.mean(x)
                datos_producto["Tendencia"] = m * x + b

         
            fig, ax = plt.subplots(figsize=(10, 6.2))
            ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], label='Unidades Vendidas', color='blue')

          
            if len(x) > 1:
                ax.plot(datos_producto['Fecha'], datos_producto["Tendencia"], label='Tendencia', color='red', linestyle='--')

           
            for fecha in datos_producto['Fecha'].iloc[::2]: 
                ax.axvline(fecha, color='black', linestyle='-', alpha=0.5)
 
            ax.set_title(f"Evolución de Ventas ({producto})")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Unidades Vendidas")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)


def main():
    st.set_page_config(layout="wide", page_title="Análisis de Ventas")
    st.title("Análisis de Ventas de Productos")
     
    mostrar_informacion_alumno()
    
    datos = cargar_datos()

    if datos is not None:
        datos_filtrados, sucursal_seleccionada = filtrar_por_sucursal(datos)
        mostrar_graficos(datos_filtrados, sucursal_seleccionada)

if __name__ == "__main__":
    main()