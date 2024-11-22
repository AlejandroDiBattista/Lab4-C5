import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCIÓN: Debe colocar la dirección en la que ha sido publicada la aplicación en la siguiente línea
# url = 'https://gabriell-v16-parcial-labb4-c5-ejercicio-izfnvv.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.899')
        st.markdown('**Nombre:** Villagra Juan Gabriel')
        st.markdown('**Comisión:** C5')

st.set_page_config(page_title="Análisis de Ventas", layout="wide")

st.markdown("""
    <style>
    .positivo { color: #28a745; }
    .negativo { color: #dc3545; }
    .contenedor-metrica {
        display: flex;
        flex-direction: column;
        margin-bottom: 1.5rem;
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .valor-metrica { 
        font-size: 2.5rem; 
        font-weight: bold;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 0.5rem 0;
        color: white;
    }
    .etiqueta-metrica { 
        font-size: 1rem; 
        color: #888;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        margin-bottom: 0.25rem;
    }
    .cambio-metrica {
        font-size: 1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

def calcular_metricas(df, producto, df_periodo_anterior=None):
    datos_producto = df[df['Producto'] == producto]
    
    ingreso_total = datos_producto['Ingreso_total'].sum()
    unidades_totales = datos_producto['Unidades_vendidas'].sum()
    precio_promedio = ingreso_total / unidades_totales if unidades_totales > 0 else 0
    
    costo_total = datos_producto['Costo_total'].sum()
    margen_promedio = ((ingreso_total - costo_total) / ingreso_total * 100) if ingreso_total > 0 else 0
    
    unidades_vendidas = unidades_totales
    
    if df_periodo_anterior is not None:
        datos_producto_anterior = df_periodo_anterior[df_periodo_anterior['Producto'] == producto]
        ingreso_total_anterior = datos_producto_anterior['Ingreso_total'].sum()
        unidades_totales_anterior = datos_producto_anterior['Unidades_vendidas'].sum()
        precio_promedio_anterior = ingreso_total_anterior / unidades_totales_anterior if unidades_totales_anterior > 0 else 0
        margen_promedio_anterior = ((ingreso_total_anterior - datos_producto_anterior['Costo_total'].sum()) / ingreso_total_anterior * 100) if ingreso_total_anterior > 0 else 0
        
        cambio_precio = ((precio_promedio - precio_promedio_anterior) / precio_promedio_anterior * 100) if precio_promedio_anterior > 0 else 0
        cambio_margen = margen_promedio - margen_promedio_anterior
        cambio_unidades = ((unidades_vendidas - unidades_totales_anterior) / unidades_totales_anterior * 100) if unidades_totales_anterior > 0 else 0
    else:
        cambio_precio = cambio_margen = cambio_unidades = 0
    
    return precio_promedio, margen_promedio, unidades_vendidas, cambio_precio, cambio_margen, cambio_unidades

def mostrar_metrica(etiqueta, valor, cambio, tipo_formato="numero"):
    if tipo_formato == "moneda":
        valor_formateado = f"${valor:,.0f}"
    elif tipo_formato == "porcentaje":
        valor_formateado = f"{valor:.0f}%"
    else:
        valor_formateado = f"{valor:,.0f}"
    
    clase_cambio = 'positivo' if cambio >= 0 else 'negativo'
    flecha_cambio = '↑' if cambio > 0 else '↓'
    signo_cambio = '+' if cambio > 0 else '-'
    valor_cambio = f"{signo_cambio}{abs(cambio):.2f}%"
    
    st.markdown(f"""
    <div class="contenedor-metrica">
        <div class="etiqueta-metrica">{etiqueta}</div>
        <div class="valor-metrica">{valor_formateado}</div>
        <div class="cambio-metrica {clase_cambio}">
            {flecha_cambio} {valor_cambio}
        </div>
    </div>
    """, unsafe_allow_html=True)

def calcular_linea_tendencia(x, y):
    coeffs = np.polyfit(x, y, 1)
    pendiente = coeffs[0]
    intercepto = coeffs[1]
    linea_tendencia = pendiente * x + intercepto
    return linea_tendencia

st.sidebar.title("Cargar archivo de datos")
st.sidebar.write("Subir archivo CSV")
archivo_cargado = st.sidebar.file_uploader("Arrastrar y soltar archivo aquí", type=['csv'])
sucursales = ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

if not archivo_cargado:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
else:
    df = pd.read_csv(archivo_cargado)
    
    anio_actual = df['Año'].max()
    anio_anterior = anio_actual - 1
    
    df_actual = df[df['Año'] == anio_actual]
    df_anterior = df[df['Año'] == anio_anterior]
    
    if sucursal_seleccionada != "Todas":
        df_actual = df_actual[df_actual['Sucursal'] == sucursal_seleccionada]
        df_anterior = df_anterior[df_anterior['Sucursal'] == sucursal_seleccionada]
    
    st.markdown(f"""
        <h1 style="color: white; font-family: 'Inter', sans-serif; margin-bottom: 2rem;">
            Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}
        </h1>
    """, unsafe_allow_html=True)
    
    for producto in df_actual['Producto'].unique():
        col_metricas, col_grafico = st.columns([1, 2])

        with col_metricas:
            st.header(producto)
            precio_promedio, margen_promedio, unidades_vendidas, cambio_precio, cambio_margen, cambio_unidades = calcular_metricas(df_actual, producto, df_anterior)
            
            mostrar_metrica("Precio Promedio", precio_promedio, cambio_precio, "moneda")
            mostrar_metrica("Margen Promedio", margen_promedio, cambio_margen, "porcentaje")
            mostrar_metrica("Unidades Vendidas", unidades_vendidas, cambio_unidades)

        with col_grafico:
            st.write("")
            st.write("")  
            st.write("")  
            st.write("")  
            st.write("")  
            st.write("")  
            st.write("")  
            st.write("")  
            st.write("") 
            st.write("")  
            st.write("")  
             

            serie_temporal = df[df['Producto'] == producto].groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
            serie_temporal['Fecha'] = pd.to_datetime(serie_temporal['Año'].astype(str) + '-' + 
                                                serie_temporal['Mes'].astype(str).str.zfill(2) + '-01')
            serie_temporal = serie_temporal.sort_values('Fecha')
            
            x = np.arange(len(serie_temporal))
            y = serie_temporal['Unidades_vendidas'].values
            linea_tendencia = calcular_linea_tendencia(x, y)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(serie_temporal['Fecha'], serie_temporal['Unidades_vendidas'], label=producto)
            ax.plot(serie_temporal['Fecha'], linea_tendencia, label='Tendencia', linestyle='--')
            
            ax.set_title("Evolución de Ventas Mensual")
            ax.set_xlabel("Año-Mes")
            ax.set_ylabel("Unidades Vendidas")
            ax.legend()
            
            plt.xticks(rotation=45)
            
            # Ajustar el diseño
            plt.tight_layout()
            
            st.pyplot(fig)

mostrar_informacion_alumno()