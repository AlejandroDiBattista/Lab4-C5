import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.899')
        st.markdown('**Nombre:** Villagra Juan Gabriel')
        st.markdown('**Comisión:** C5')

st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

def calcular_metricas(data_ventas):
    data_ventas['precio_unitario'] = data_ventas['Ingreso_total'] / data_ventas['Unidades_vendidas']
    data_ventas['utilidad'] = data_ventas['Ingreso_total'] - data_ventas['Costo_total']
    data_ventas['rentabilidad'] = (data_ventas['utilidad'] / data_ventas['Ingreso_total']) * 100

    precio_unitario_promedio = data_ventas['precio_unitario'].mean()
    rentabilidad_promedio = data_ventas['rentabilidad'].mean()
    total_unidades = data_ventas['Unidades_vendidas'].sum()

    metricas_anuales = {
        'precios': data_ventas.groupby('Año')['precio_unitario'].mean(),
        'rentabilidad': data_ventas.groupby('Año')['rentabilidad'].mean(),
        'volumen': data_ventas.groupby('Año')['Unidades_vendidas'].sum()
    }

    def calcular_tendencia(serie_temporal):
        if len(serie_temporal) > 1:
            return serie_temporal.pct_change().mean() * 100
        return 0

    tendencias = {
        'precio': calcular_tendencia(metricas_anuales['precios']),
        'rentabilidad': calcular_tendencia(metricas_anuales['rentabilidad']),
        'volumen': calcular_tendencia(metricas_anuales['volumen'])
    }

    return {
        "precio_promedio": round(precio_unitario_promedio, 2),
        "variacion_precio_promedio_anual": round(tendencias['precio'], 2),
        "margen_promedio": round(rentabilidad_promedio, 2),
        "variacion_margen_promedio_anual": round(tendencias['rentabilidad'], 2),
        "unidades_vendidas": int(total_unidades),
        "variacion_anual_unidades": round(tendencias['volumen'], 2),
    }

def crear_grafico_ventas(data_ventas, nombre_producto):
    ventas_mensuales = data_ventas.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(ventas_mensuales)), ventas_mensuales['Unidades_vendidas'], 
            label=nombre_producto, linewidth=2)

    x_valores = np.arange(len(ventas_mensuales))
    coef_tendencia = np.polyfit(x_valores, ventas_mensuales['Unidades_vendidas'], 1)
    linea_tendencia = np.poly1d(coef_tendencia)
    ax.plot(x_valores, linea_tendencia(x_valores), 
            linestyle='--', color='red', label='Tendencia', linewidth=1.5)

    ax.set_title('Evolución de Ventas Mensual', pad=20)
    ax.set_xlabel(' Año - Mes ',labelpad=10)
    ax.set_ylabel('Unidades Vendidas', labelpad=10)
    ax.set_ylim(0,None)
    
    marcas_tiempo = [f"{row.Año}" if row.Mes == 1 else "" 
                    for row in ventas_mensuales.itertuples()]
    ax.set_xticks(range(len(ventas_mensuales)))
    ax.set_xticklabels(marcas_tiempo,)
    
    ax.legend(title='Referencia')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

st.sidebar.header("Análisis de Datos")
archivo_datos = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])

if archivo_datos is not None:
    dataset = pd.read_csv(archivo_datos)
    lista_sucursales = ["Todas"] + dataset['Sucursal'].unique().tolist()
    sucursal_elegida = st.sidebar.selectbox("Seleccionar Sucursal", lista_sucursales)

    if sucursal_elegida != "Todas":
        dataset = dataset[dataset['Sucursal'] == sucursal_elegida]
        st.title(f"Análisis de {sucursal_elegida}")
    else:
        st.title("Datos de Todas las Sucursales")

    for producto in dataset['Producto'].unique():
        with st.container(border=True):
            st.subheader(producto)
            datos_producto = dataset[dataset['Producto'] == producto]
            indicadores = calcular_metricas(datos_producto)

            col1, col2 = st.columns([0.25, 0.75])

            with col1:
                st.metric(
                    label="Precio Unitario Promedio", 
                    value=f"${indicadores['precio_promedio']:,.0f}".replace(",", "."),
                    delta=f"{indicadores['variacion_precio_promedio_anual']:.2f}%"
                )
                st.metric(
                    label="Rentabilidad Promedio", 
                    value=f"{indicadores['margen_promedio']:.0f}%".replace(",", "."),
                    delta=f"{indicadores['variacion_margen_promedio_anual']:.2f}%"
                )
                st.metric(
                    label="Volumen de Ventas", 
                    value=f"{indicadores['unidades_vendidas']:,.0f}".replace(",", "."),
                    delta=f"{indicadores['variacion_anual_unidades']:.2f}%"
                )

            with col2:
                grafico = crear_grafico_ventas(datos_producto, producto)
                st.pyplot(grafico)
else:
    st.subheader("Por favor, cargue un archivo CSV")

mostrar_informacion_alumno()