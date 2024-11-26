import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://fedetouceda.streamlit.app'


st.set_page_config(page_title="Ventas de Sucursales", layout="wide")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 57.543')
        st.markdown('**Nombre:** Touceda Federico')
        st.markdown('**Comisión:** C5')
mostrar_informacion_alumno()


st.markdown(
    """
    <style>
    .metric-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .metric-container .stMetric {
        margin: 0; 
    }
    .stPlotlyChart, .stPyplot {
        padding: 0; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


def calcular_tendencia(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p(x)


def crear_grafico_ventas(datos_producto, producto):
    ventas_por_fecha = datos_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    ventas_por_fecha = ventas_por_fecha.sort_values('Fecha')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ventas_por_fecha['Fecha'], ventas_por_fecha['Unidades_vendidas'], label=producto)
    x = ventas_por_fecha['Fecha'].map(lambda date: date.toordinal())
    y = ventas_por_fecha['Unidades_vendidas']
    tendencia = calcular_tendencia(x, y)
    ax.plot(ventas_por_fecha['Fecha'], tendencia, linestyle='--', color='red', label='Tendencia')
    ax.set_title(f'Evolución de Ventas Mensual - {producto}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Unidades Vendidas')
    
    
    if producto == "Coca Cola":
        ax.set_ylim(0, 50000)  
        ax.set_yticks(range(0, 50001, 10000))  
    
    ax.legend()
    ax.grid(True)
    plt.tight_layout()  
    return fig


st.sidebar.header("Cargar archivo de datos")
cargar_archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if cargar_archivo is not None:
    data = pd.read_csv(cargar_archivo)
    
    
    data = data.rename(columns={'Año': 'year', 'Mes': 'month'})
    
    
    data['Fecha'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    
    
    sucursales = ["Todas"] + data['Sucursal'].unique().tolist()
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    
    if sucursal != "Todas":
        data = data[data['Sucursal'] == sucursal]
        st.title(f"Datos de {sucursal}")
    else:
        st.title("Datos de Todas las Sucursales")
    
    
    productos = data['Producto'].unique()
    for producto in productos:
        with st.container(border=True):
            st.subheader(f"{producto}")
            datos_producto = data[data['Producto'] == producto].copy()
            
            
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            precio_promedio = round(datos_producto['Precio_promedio'].mean())
            
            precio_promedio_anual = datos_producto.groupby('year')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = round(precio_promedio_anual.pct_change().mean() * 100, 2)
            
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100
           
            margen_promedio = round(datos_producto['Margen'].mean())
            margen_promedio_anual = datos_producto.groupby('year')['Margen'].mean()
            variacion_margen_promedio_anual = round(margen_promedio_anual.pct_change().mean() * 100, 2)
            
            unidades_promedio = round(datos_producto['Unidades_vendidas'].mean())
            unidades_vendidas = round(datos_producto['Unidades_vendidas'].sum())
            
           
            unidades_por_año = datos_producto.groupby('year')['Unidades_vendidas'].sum()
            variacion_anual_unidades = round(unidades_por_año.pct_change().mean() * 100, 2)
            
            
            col1, col2 = st.columns([1, 2])  
            
           
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(
                    label="Precio Promedio",
                    value=f"${precio_promedio:,}".replace(",", "."),
                    delta=f"{variacion_precio_promedio_anual}%"
                )
                st.metric(
                    label="Margen Promedio",
                    value=f"{margen_promedio}%",
                    delta=f"{variacion_margen_promedio_anual}%"
                )
                st.metric(
                    label="Unidades Vendidas",
                    value=f"{unidades_vendidas:,}".replace(",", "."),
                    delta=f"{variacion_anual_unidades}%"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            
            with col2:
                fig = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(fig)
else:
    st.info("Cargue un archivo CSV para comenzar.")