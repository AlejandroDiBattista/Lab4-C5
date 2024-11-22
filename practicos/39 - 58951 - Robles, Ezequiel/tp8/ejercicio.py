import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://tp8ezequiel-robles-58951-k5dw.streamlit.app/'

def mostrar_informacion_alumno():
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
   
    with st.container(border=True):
        st.markdown('**Legajo:** 58.951')
        st.markdown('**Nombre:** Ezequiel Robles')
        st.markdown('**Comisión:** C5')

st.sidebar.title("Cargar archivo de datos")
st.sidebar.write("Sube tu archivo CSV para analizar los datos.")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is None:
    mostrar_informacion_alumno()

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        data.columns = data.columns.str.strip()

        expected_columns = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
        if set(expected_columns).issubset(data.columns):

            data['Día'] = 1

            data.rename(columns={'Año': 'year', 'Mes': 'month', 'Día': 'day'}, inplace=True)

            data['Fecha'] = pd.to_datetime(data[['year', 'month', 'day']], errors='coerce')

            data.rename(columns={'year': 'Año', 'month': 'Mes', 'day': 'Día'}, inplace=True)

            data = data.dropna(subset=['Fecha'])

            sucursal_seleccionada = st.sidebar.selectbox("Selecciona una Sucursal", ["Todas"] + list(data['Sucursal'].unique()))

            if sucursal_seleccionada != "Todas":
                data = data[data['Sucursal'] == sucursal_seleccionada]

            st.title(f"Datos de {sucursal_seleccionada if sucursal_seleccionada != 'Todas' else 'Todas las Sucursales'}")

            for producto in data['Producto'].unique():
                
             with st.container(border=True):
                producto_data = data[data['Producto'] == producto]

                producto_data = producto_data.sort_values(by='Fecha')

                precio_promedio = producto_data['Ingreso_total'].sum() / producto_data['Unidades_vendidas'].sum()
                margen_promedio = (producto_data['Ingreso_total'].sum() - producto_data['Costo_total'].sum()) / producto_data['Ingreso_total'].sum()
                unidades_vendidas = producto_data['Unidades_vendidas'].sum()

                if len(producto_data) > 1:  
                    producto_data['Variacion_Precio'] = producto_data['Ingreso_total'].pct_change().fillna(0) * 100
                    producto_data['Variacion_Margen'] = (producto_data['Ingreso_total'] - producto_data['Costo_total']).pct_change().fillna(0) * 100
                    producto_data['Variacion_Unidades'] = producto_data['Unidades_vendidas'].pct_change().fillna(0) * 100
                    variacion_precio = producto_data['Variacion_Precio'].iloc[-1]
                    variacion_margen = producto_data['Variacion_Margen'].iloc[-1]
                    variacion_unidades = producto_data['Variacion_Unidades'].iloc[-1]
                else:
                    variacion_precio = 0
                    variacion_margen = 0
                    variacion_unidades = 0

                st.title(f"{producto}")

                col1, col2 = st.columns([2, 4])

                with col1:
                    st.metric("Precio Promedio", f"${precio_promedio:.3f}", f"{variacion_precio:+.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:.0%}", f"{variacion_margen:+.2f}%")
                    st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}", f"{variacion_unidades:+.2f}%")

                with col2:
                    fig, ax = plt.subplots()
                    ax.plot(producto_data['Fecha'], producto_data['Unidades_vendidas'], label='Unidades Vendidas', color='blue')

                    z = np.polyfit(producto_data.index, producto_data['Unidades_vendidas'], 1)
                    p = np.poly1d(z)
                    ax.plot(producto_data['Fecha'], p(producto_data.index), "r--", label="Tendencia")

                    ax.set_title(f"Evolución de Ventas Mensual")
                    ax.set_xlabel("Fecha")
                    ax.set_ylabel("Unidades Vendidas")
                    ax.legend()

                    st.pyplot(fig)

                

        else:
            st.error(f"El archivo cargado no contiene las columnas esperadas. Asegúrate de que el archivo tenga las columnas: {', '.join(expected_columns)}.")
    except Exception as e:
        st.error(f"Ha ocurrido un error al procesar el archivo: {e}")

