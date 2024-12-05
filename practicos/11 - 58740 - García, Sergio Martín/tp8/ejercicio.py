import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://martingarcia1-tp-final-de-lab4-tp8ejercicio-r0gymh.streamlit.app/'

orden_productos = ["Coca Cola", "Fanta", "Sprite", "7 Up", "Pepsi"]

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown("**Legajo:** 58.740")
        st.markdown("**Nombre:** García Sergio Martín")
        st.markdown("**Comisión:** C5")

def crear_grafico_ventas(data, producto):
    data['Fecha'] = pd.to_datetime(data['Año'].astype(str) + '-' + data['Mes'].astype(str) + '-01')
    ventas_mensuales = data.groupby('Fecha')['Unidades_vendidas'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(
        ventas_mensuales.index,  
        ventas_mensuales,  
        label=f"{producto}", 
        color="#1f77b4", 
        linewidth=2
    )

    x = np.arange(len(ventas_mensuales))  
    y = ventas_mensuales.values  
    z = np.polyfit(x, y, 1)  
    p = np.poly1d(z) 
    ax.plot(ventas_mensuales.index, p(x), linestyle="--", color="red", label="Tendencia", linewidth=2)  

    max_y = ventas_mensuales.max()  
    step_y = 10000  
    max_y_adjusted = (np.ceil(max_y / step_y) * step_y) if max_y > 0 else step_y  
    ax.set_ylim(0, max_y_adjusted) 
    ax.set_yticks(np.arange(0, max_y_adjusted + step_y, step_y))

    ax.set_title(f"Evolución de Ventas Mensuales - {producto}", fontsize=16)
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(loc="upper left")  
    ax.grid(alpha=0.3)  
    plt.xticks(rotation=0)  

    return fig

st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado is not None:
    datos = pd.read_csv(archivo_cargado)

    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()

    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if sucursal_seleccionada != "Todas":
        datos = datos[datos['Sucursal'] == sucursal_seleccionada]
        st.title(f"Datos de {sucursal_seleccionada}")
    else:
        st.title("Datos de Todas las Sucursales")


    # productos = [p for p in orden_productos if p in datos['Producto'].unique()]
    productos = datos['Producto'].unique()

    for producto in productos:
        with st.container(border=True):
            st.subheader(f"{producto}")
            datos_producto = datos[datos['Producto'] == producto]

            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = datos_producto['Precio_promedio'].mean()

            precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
            margen_promedio = round(datos_producto['Margen'].mean())

            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()
            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                st.metric(label="Precio Promedio", value=f"${precio_promedio:,.0f}", delta=f"{variacion_precio_promedio_anual:.2f}%")
                st.metric(label="Margen Promedio", value=f"{margen_promedio:.0f}%", delta=f"{variacion_margen_promedio_anual:.2f}%")
                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}", delta=f"{variacion_anual_unidades:.2f}%")
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    mostrar_informacion_alumno()