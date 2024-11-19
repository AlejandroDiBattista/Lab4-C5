import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

@st.cache_data
def calcular_cambio_anual(data, columna):
    if columna in data.columns:
        cambio_porcentual = data.groupby('Año')[columna].sum().pct_change()
        return cambio_porcentual.mean() * 100 if not cambio_porcentual.isna().all() else 0
    else:
        return 0

@st.cache_resource
def crear_grafico_ventas(datos_producto, producto):
    ventas_por_producto = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(ventas_por_producto)), ventas_por_producto['Unidades_vendidas'], label=producto)
    
    x = np.arange(len(ventas_por_producto))
    y = ventas_por_producto['Unidades_vendidas']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    ax.plot(x, p(x), linestyle='--', color='red', label='Tendencia')
    
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.set_xticks(range(len(ventas_por_producto)))
    
    valores = []
    for i, row in enumerate(ventas_por_producto.itertuples()):
        if row.Mes == 1:
            valores.append(f"{row.Año}")
        else:
            valores.append("")
    ax.set_xticklabels(valores, rotation=0)
    ax.set_ylim(0, None)
    ax.legend(title='Producto')
    ax.grid(True)
    
    return fig

def mostrar_informacion_alumno():
    with st.expander("", expanded=True):
        st.markdown('**Legajo:** 55.578')
        st.markdown('**Nombre:** Patricio Fernández Dumit')
        st.markdown('**Comisión:** C5')

st.sidebar.header("Cargar archivo de datos")
archivo_subido = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_subido is not None:
    datos = cargar_datos(archivo_subido)
    
    sucursales = ["Todas"] + datos['Sucursal'].unique().tolist()
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    if sucursal != "Todas":
        datos = datos[datos['Sucursal'] == sucursal]
        st.title(f"Datos de {sucursal}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    productos = datos['Producto'].unique()

    for producto in productos:
        st.markdown("---")  
        st.markdown(f"### {producto}")  
        datos_producto = datos[datos['Producto'] == producto]
        
        valores_producto = datos_producto.groupby('Año').agg(
            ingreso_total=('Ingreso_total', 'sum'),
            unidades_vendidas=('Unidades_vendidas', 'sum'),
            costo_total=('Costo_total', 'sum')
        ).reset_index()
        
        valores_producto['Precio_promedio'] = valores_producto['ingreso_total'] / valores_producto['unidades_vendidas']
        valores_producto['Ganancia'] = valores_producto['ingreso_total'] - valores_producto['costo_total']
        valores_producto['Margen'] = (valores_producto['Ganancia'] / valores_producto['ingreso_total']) * 100
        
        cambio_precio_promedio_anual = calcular_cambio_anual(valores_producto, 'Precio_promedio')
        cambio_margen_promedio_anual = calcular_cambio_anual(valores_producto, 'Margen')
        cambio_unidades_vendidas_anual = calcular_cambio_anual(valores_producto, 'unidades_vendidas')
        
        col1, col2 = st.columns([0.25, 0.75])
        
        with col1:
            st.metric(label="Precio Promedio", value=f"${valores_producto['Precio_promedio'].mean():,.0f}".replace(",", "."), delta=f"{cambio_precio_promedio_anual:.2f}%")
            st.metric(label="Margen Promedio", value=f"{valores_producto['Margen'].mean():,.0f}%".replace(",", "."), delta=f"{cambio_margen_promedio_anual:.2f}%")
            st.metric(label="Unidades Vendidas", value=f"{valores_producto['unidades_vendidas'].sum():,.0f}".replace(",", "."), delta=f"{cambio_unidades_vendidas_anual:.2f}%")
        
        with col2:
            fig = crear_grafico_ventas(datos_producto, producto)
            st.pyplot(fig)
else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    mostrar_informacion_alumno()