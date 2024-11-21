import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-58773.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.773')
        st.markdown('**Nombre:** Elli Salazar Geronimo')
        st.markdown('**Comisión:** C5')

# Mostrar información del alumno
mostrar_informacion_alumno()


st.title("Datos de Todas las Sucursales")


st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])


sucursal_seleccionada = st.sidebar.selectbox(
    "Seleccionar Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
)


if uploaded_file is not None:
   
    datos = pd.read_csv(uploaded_file)
    

    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]
    
  
    productos = datos["Producto"].unique()
    for producto in productos:
        datos_producto = datos[datos["Producto"] == producto]
        
    
        precio_promedio = datos_producto["Ingreso_total"].sum() / datos_producto["Unidades_vendidas"].sum()
        margen_promedio = (
            (datos_producto["Ingreso_total"].sum() - datos_producto["Costo_total"].sum())
            / datos_producto["Ingreso_total"].sum()
        )
        unidades_vendidas = datos_producto["Unidades_vendidas"].sum()
        
     
        st.subheader(producto)
        st.metric("Precio Promedio", f"${precio_promedio:,.3f}")
        st.metric("Margen Promedio", f"{margen_promedio * 100:.0f}%")
        st.metric("Unidades Vendidas", f"{unidades_vendidas:,}")
        
       
        datos_producto["Fecha"] = pd.to_datetime(
            datos_producto["Año"].astype(str) + "-" + datos_producto["Mes"].astype(str) + "-01"
        )
        datos_agrupados = datos_producto.groupby("Fecha").sum().reset_index()
        
  
        x = np.arange(len(datos_agrupados)).reshape(-1, 1)
        y = datos_agrupados["Unidades_vendidas"].values
        modelo = LinearRegression()
        modelo.fit(x, y)
        tendencia = modelo.predict(x)
        
   
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(datos_agrupados["Fecha"], datos_agrupados["Unidades_vendidas"], label="Unidades Vendidas")
        ax.plot(datos_agrupados["Fecha"], tendencia, label="Tendencia", color="red")
        ax.set_title(f"Evolución de Ventas Mensual - {producto}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Unidades Vendidas")
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Por favor, sube un archivo CSV para comenzar.")
