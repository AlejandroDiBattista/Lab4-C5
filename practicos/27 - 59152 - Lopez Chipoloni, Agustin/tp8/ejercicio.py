import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Datos de Todas las Sucursales")


st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)
    expected_columns = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
    
    if all(col in data.columns for col in expected_columns):
        sucursales = ["Todas"] + list(data["Sucursal"].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
        if sucursal_seleccionada != "Todas":
            data = data[data["Sucursal"] == sucursal_seleccionada]

        data["Precio_promedio"] = data["Ingreso_total"] / data["Unidades_vendidas"]
        data["Margen_promedio"] = (data["Ingreso_total"] - data["Costo_total"]) / data["Ingreso_total"]
        
        st.header(f"Datos de {sucursal_seleccionada}")
        for producto in data["Producto"].unique():
            producto_data = data[data["Producto"] == producto]
            
            precio_promedio = producto_data["Precio_promedio"].mean()
            margen_promedio = producto_data["Margen_promedio"].mean()
            unidades_vendidas = producto_data["Unidades_vendidas"].sum()
    
            unidades_tendencia = producto_data["Unidades_vendidas"].pct_change().mean() * 100
            precio_tendencia = producto_data["Precio_promedio"].pct_change().mean() * 100
            margen_tendencia = producto_data["Margen_promedio"].pct_change().mean() * 100

            with st.container():
                st.subheader(producto)
                col1, col2 = st.columns([1, 2])        
                with col1:
                    st.metric(
                        "Precio Promedio",
                        f"${precio_promedio:.2f}",
                        f"{precio_tendencia:.2f}%",
                        delta_color="inverse" if precio_tendencia < 0 else "normal",
                    )
                    st.metric(
                        "Margen Promedio",
                        f"{margen_promedio:.2%}",
                        f"{margen_tendencia:.2f}%",
                        delta_color="normal",
                    )
                    st.metric(
                        "Unidades Vendidas",
                        f"{int(unidades_vendidas):,}",
                        f"{unidades_tendencia:.2f}%",
                        delta_color="normal",
                    )
                with col2:
                    fig, ax = plt.subplots()
                    producto_data["Fecha"] = pd.to_datetime(
                        producto_data["Año"].astype(str) + '-' + producto_data["Mes"].astype(str) + '-1'
                    )
                    producto_data = producto_data.sort_values("Fecha")

                    ax.plot(
                        producto_data["Fecha"], 
                        producto_data["Unidades_vendidas"], 
                        label=f"{producto}", 
                        marker="o",
                    )
                    z = np.polyfit(range(len(producto_data)), producto_data["Unidades_vendidas"], 1)
                    p = np.poly1d(z)
                    ax.plot(
                        producto_data["Fecha"], 
                        p(range(len(producto_data))), 
                        label="Tendencia", 
                        linestyle="--", 
                        color="red",
                    )

                    ax.set_title("Evolución de Ventas Mensual")
                    ax.set_xlabel("Año-Mes")
                    ax.set_ylabel("Unidades Vendidas")
                    ax.legend()
                    st.pyplot(fig)
    else:
        st.error("El archivo cargado no tiene las columnas esperadas.")
else:
    st.info("Por favor, suba un archivo CSV para comenzar.")



             



## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59152')
        st.markdown('**Nombre:** Lopez Chipolo Nahuel Agustin')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()