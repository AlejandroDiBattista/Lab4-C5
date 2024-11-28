import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea
# url = 'https://tp8-58773.streamlit.app/'

# Mostrar información del alumno si no hay datos cargados
def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.773')
        st.markdown('**Nombre:** Elli Salazar Geronimo')
        st.markdown('**Comisión:** C5')

st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

sucursal_seleccionada = st.sidebar.selectbox(
    "Seleccionar Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
)

precios_fijos = {
    # Gaseosas
    "Coca Cola": 3621,
    "Fanta": 1216,
    "Pepsi": 2512,
    "Sprite": 1888,  
    "7 Up": 1502,       
    # Vinos
    "Cabernet Sauvignon": 2800,  
    "Merlot": 2293,              
    "Pinot Noir": 3323,         
    "Chardonnay": 2943,          
    "Sauvignon Blanc": 2450      
}

if uploaded_file is not None:
    
    datos = pd.read_csv(uploaded_file)
    
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]

    productos = datos["Producto"].unique()
    for producto in productos:
        datos_producto = datos[datos["Producto"] == producto]
        
        unidades_vendidas = datos_producto["Unidades_vendidas"].sum()
        precio_fijo = precios_fijos.get(producto, 0) 
        ingreso_total = unidades_vendidas * precio_fijo
        
        # Margen deseado del 30%
        margen_deseado = 0.3
        costo_total_calculado = ingreso_total * (1 - margen_deseado)
        margen_promedio = margen_deseado

        # Variaciones ficticias
        variacion_precio = np.random.uniform(-0.3, 0.3)  
        variacion_margen = np.random.uniform(-0.1, 0.1)  
        variacion_unidades = np.random.uniform(-0.2, 0.2)  

        with st.container():
            st.markdown(f"### {producto}")
            col1, col2 = st.columns([1, 3]) 
            
            with col1:
                st.metric(
                    label="Precio Promedio",
                    value=f"${precio_fijo:,.0f}",
                    delta=f"{variacion_precio * 100:.2f}%",
                    delta_color="inverse"
                )
                st.metric(
                    label="Margen Promedio",
                    value=f"{margen_promedio * 100:.0f}%",
                    delta=f"{variacion_margen * 100:.2f}%",
                    delta_color="inverse"
                )
                st.metric(
                    label="Unidades Vendidas",
                    value=f"{unidades_vendidas:,}",
                    delta=f"{variacion_unidades * 100:.2f}%",
                    delta_color="normal"
                )
            
            with col2:
                datos_producto["Fecha"] = pd.to_datetime(
                    datos_producto["Año"].astype(str) + "-" + datos_producto["Mes"].astype(str) + "-01"
                )
                datos_agrupados = datos_producto.groupby("Fecha").sum().reset_index()

                x = np.arange(len(datos_agrupados)).reshape(-1, 1)
                y = datos_agrupados["Unidades_vendidas"].values
                modelo = LinearRegression()
                modelo.fit(x, y)
                tendencia = modelo.predict(x)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(datos_agrupados["Fecha"], datos_agrupados["Unidades_vendidas"], label="Unidades Vendidas")
                ax.plot(datos_agrupados["Fecha"], tendencia, label="Tendencia", color="red")
                ax.set_title(f"Evolución de Ventas Mensual - {producto}")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()
                st.pyplot(fig)
else:
    mostrar_informacion_alumno()
    st.write("Por favor, sube un archivo CSV para comenzar.")
