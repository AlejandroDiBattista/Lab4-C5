import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL de la aplicación publicada
url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.sidebar:
        st.markdown('### Información del Alumno')
        st.markdown('**Legajo:** 58866')
        st.markdown('**Nombre:** Velazquez, Tobias Alejo')
        st.markdown('**Comisión:** C5')

def cargar_datos():
    with st.sidebar:
        st.header("📂 Cargar archivo de datos")
        archivo = st.file_uploader("Subir archivo CSV", type=["csv"])
        if archivo is not None:
            datos = pd.read_csv(archivo)
            return datos
    return None

def calcular_indicadores(datos):
    datos['Precio_Promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    datos['Margen_Promedio'] = (datos['Ingreso_total'] - datos['Costo_total']) / datos['Ingreso_total']
    return datos

def graficar_evolucion(datos, producto):
    datos_producto = datos[datos['Producto'] == producto]
    datos_agrupados = datos_producto.groupby(['Año', 'Mes']).sum().reset_index()
    
    plt.figure(figsize=(6, 3))
    plt.plot(
        datos_agrupados['Mes'], 
        datos_agrupados['Unidades_vendidas'], 
        marker='o', label=f"{producto}"
    )
    z = np.polyfit(datos_agrupados['Mes'], datos_agrupados['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    plt.plot(datos_agrupados['Mes'], p(datos_agrupados['Mes']), "r--", label="Tendencia")
    plt.title(f"Evolución de Ventas Mensual")
    plt.xlabel("Mes")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    st.pyplot(plt)

def mostrar_informacion(datos):
    sucursales = ["Todas"] + list(datos['Sucursal'].unique())
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal != "Todas":
        datos = datos[datos['Sucursal'] == sucursal]
    
    st.header(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")
    productos = datos['Producto'].unique()
    
    for producto in productos:
        # Estilo CSS para la caja
        st.markdown("""
            <style>
            .custom-box {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                background-color: #fff;
            }
            </style>
        """, unsafe_allow_html=True)

        # Contenedor para los datos del producto
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.markdown(f"### **{producto}**")
            producto_datos = datos[datos['Producto'] == producto]
            
            precio_promedio = producto_datos['Precio_Promedio'].mean()
            margen_promedio = producto_datos['Margen_Promedio'].mean()
            unidades_vendidas = producto_datos['Unidades_vendidas'].sum()
            
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}")
            st.metric("Margen Promedio", f"{margen_promedio:.0%}")
            st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}")

        with col2:
            graficar_evolucion(datos, producto)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    mostrar_informacion_alumno()
    
    datos = cargar_datos()
    if datos is not None:
        datos = calcular_indicadores(datos)
        mostrar_informacion(datos)

if __name__ == "__main__":
    main()
