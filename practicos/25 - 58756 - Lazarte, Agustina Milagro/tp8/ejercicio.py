import streamlit as st
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mostrar_informacion_alumno():
    with st.container():
        st.markdown("### Por favor, sube un archivo CSV desde la barra lateral.")
        st.markdown("#### **Legajo:** 58756")
        st.markdown("#### **Nombre:** Agustina Milagro Lazarte")
        st.markdown("#### **Comisión:** C5")

        mostrar_informacion_alumno()


st.title("Datos de todas las sucursales")


st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)

   
    data['Precio_promedio'] = data['Ingreso_total'] / data['Unidades_vendidas']
    data['Margen_promedio'] = (data['Ingreso_total'] - data['Costo_total']) / data['Ingreso_total']

   
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + list(data['Sucursal'].unique()))

   
    if sucursal != "Todas":
        data = data[data['Sucursal'] == sucursal]


    resumen = data.groupby('Producto').agg(
        Precio_promedio=pd.NamedAgg(column='Precio_promedio', aggfunc='mean'),
        Margen_promedio=pd.NamedAgg(column='Margen_promedio', aggfunc='mean'),
        Unidades_vendidas=pd.NamedAgg(column='Unidades_vendidas', aggfunc='sum')
    ).reset_index()

  
    for producto in resumen['Producto'].unique():
        st.subheader(producto)

      
        prod_data = data[data['Producto'] == producto]
        prod_resumen = resumen[resumen['Producto'] == producto]

      
        col1, col2, col3 = st.columns(3)
        col1.metric("Precio Promedio", f"${prod_resumen['Precio_promedio'].values[0]:,.2f}")
        col2.metric("Margen Promedio", f"{prod_resumen['Margen_promedio'].values[0]*100:.2f}%")
        col3.metric("Unidades Vendidas", f"{int(prod_resumen['Unidades_vendidas'].values[0]):,}")

       
        prod_data['Fecha'] = pd.to_datetime(
            prod_data['Año'].astype(str) + "-" + prod_data['Mes'].astype(str) + "-01"
        )

       
        prod_data = prod_data.sort_values('Fecha')

        
        plt.figure(figsize=(10, 4))
        plt.plot(prod_data['Fecha'], prod_data['Unidades_vendidas'], label="Ventas Mensuales", marker="o")
        z = np.polyfit(range(len(prod_data)), prod_data['Unidades_vendidas'], 1)
        p = np.poly1d(z)
        plt.plot(prod_data['Fecha'], p(range(len(prod_data))), "r--", label="Tendencia")
        plt.title(f"Evolución de Ventas Mensual - {producto}")
        plt.xlabel("Fecha")
        plt.ylabel("Unidades Vendidas")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)
else:
    st.info("Sube un archivo CSV para comenzar.")