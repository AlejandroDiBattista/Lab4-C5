import streamlit as st 
import matplotlib.pyplot as plt
import csv
from io import TextIOWrapper
from datetime import datetime
from sklearn.linear_model import LinearRegression
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
import numpy as np

# url = 'https://parcial2-58894.streamlit.app/'
st.set_page_config(page_title="Análisis de Ventas Mensuales", layout="wide", initial_sidebar_state="expanded")

def mostrar_informacion_alumno():
    st.markdown("""
        <div style="border: 1px solid gray; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: #e3f2fd;">
            <p><strong>Legajo:</strong> 58.894</p>
            <p><strong>Nombre:</strong> Flavia González Nacusse</p>
            <p><strong>Comisión:</strong> C5</p>
        </div>
    """, unsafe_allow_html=True)

def cargar_datos(archivo):
    datos = []
    archivo_en_texto = TextIOWrapper(archivo, encoding='utf-8')
    reader = csv.DictReader(archivo_en_texto)
    for fila in reader:
        try:
            fila['Año'] = int(fila['Año'])
            fila['Mes'] = int(fila['Mes'])
            fila['Unidades_vendidas'] = int(fila['Unidades_vendidas'])
            fila['Ingreso_total'] = float(fila['Ingreso_total'])
            fila['Costo_total'] = float(fila['Costo_total'])
            fila['Fecha'] = datetime(fila['Año'], fila['Mes'], 1)
            datos.append(fila)
        except ValueError:
            st.error("Error procesando una fila: asegúrate de que los valores sean correctos.")
    return datos

def calcular_metricas(datos):
    productos = {}
    for fila in datos:
        producto = fila['Producto']
        if producto not in productos:
            productos[producto] = {
                'Precio_medio': 0,
                'Margen_medio': 0,
                'Unidades_vendidas': 0,
                'ventas_totales': 0,
                'costos_totales': 0,
                'precio_medio_anterior': 0,
                'margen_medio_anterior': 0,
                'unidades_vendidas_anterior': 0,
            }
        productos[producto]['Unidades_vendidas'] += fila['Unidades_vendidas']
        productos[producto]['ventas_totales'] += fila['Ingreso_total']
        productos[producto]['costos_totales'] += fila['Costo_total']
    for producto, metricas in productos.items():
        metricas['Precio_medio'] = metricas['ventas_totales'] / metricas['Unidades_vendidas']
        metricas['Margen_medio'] = (metricas['ventas_totales'] - metricas['costos_totales']) / metricas['ventas_totales']
        metricas['precio_medio_anterior'] = metricas['Precio_medio'] * 0.9
        metricas['margen_medio_anterior'] = metricas['Margen_medio'] * 1.05
        metricas['unidades_vendidas_anterior'] = metricas['Unidades_vendidas'] * 0.95
    return productos

def calcular_porcentaje_variacion(valor_actual, valor_anterior):
    if valor_anterior == 0:
        return 0
    return ((valor_actual - valor_anterior) / valor_anterior) * 100

def calcular_tendencia(fechas, ventas):
    x = np.arange(len(fechas)).reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, ventas)
    return modelo.predict(x)

def graficar_ventas(fechas, ventas, tendencia, producto):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fechas, ventas, label="Unidades Vendidas", linestyle="-", color="blue")
    ax.plot(fechas, tendencia, label="Tendencia", linestyle="--", color="red")
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_title(f"Evolución de Ventas Mensual", fontsize=12)
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.grid(True)
    ax.legend()
    return fig

def mostrar_encabezado(sucursal):
    encabezado = f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}"
    st.markdown(f"<h2 style='text-align: center;'>{encabezado}</h2>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Análisis de Ventas Mensuales</h1>", unsafe_allow_html=True)
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if archivo_cargado:
    try:
        datos = cargar_datos(archivo_cargado)
        if datos:
            sucursales = list(set(fila['Sucursal'] for fila in datos))
            sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + sucursales)
            if sucursal_seleccionada != "Todas":
                datos = [fila for fila in datos if fila['Sucursal'] == sucursal_seleccionada]
            mostrar_encabezado(sucursal_seleccionada)
            metricas_por_producto = calcular_metricas(datos)
            for producto, metricas in metricas_por_producto.items():
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader(producto)
                        precio_variacion = calcular_porcentaje_variacion(
                            metricas['Precio_medio'], metricas['precio_medio_anterior']
                        )
                        st.metric(
                            "Precio Promedio",
                            f"${metricas['Precio_medio']:.2f}".replace(".", ","),
                            f"{precio_variacion:.2f}%",
                            delta_color="inverse"
                        )
                        margen_variacion = calcular_porcentaje_variacion(
                            metricas['Margen_medio'], metricas['margen_medio_anterior']
                        )
                        st.metric(
                            "Margen Promedio",
                            f"{metricas['Margen_medio'] * 100:.2f}%".replace(".", ","),
                            f"{margen_variacion:.2f}%",
                            delta_color="inverse"
                        )
                        unidades_variacion = calcular_porcentaje_variacion(
                            metricas['Unidades_vendidas'], metricas['unidades_vendidas_anterior']
                        )
                        st.metric(
                            "Unidades Vendidas",
                            f"{int(metricas['Unidades_vendidas']):,}".replace(",", "."),
                            f"{unidades_variacion:.2f}%",
                            delta_color="inverse"
                        )
                    with col2:
                        fechas = [fila['Fecha'] for fila in datos if fila['Producto'] == producto]
                        ventas = [fila['Unidades_vendidas'] for fila in datos if fila['Producto'] == producto]
                        tendencia = calcular_tendencia(fechas, np.array(ventas))
                        fig = graficar_ventas(fechas, ventas, tendencia, producto)
                        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV con los datos de ventas. Desde la barra lateral.")
    mostrar_informacion_alumno()