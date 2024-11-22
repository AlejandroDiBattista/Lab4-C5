import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://augusrojas-parcial-2-lab-tp8ejercicio-dp3hjm.streamlit.app/' 

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58.806')
        st.markdown('**Nombre:** Rojas Carlos Augusto')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()

st.sidebar.title('Cargar archivo de datos')
csv = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if csv is None:
    st.sidebar.warning("Subí un archivo CSV para comenzar.")
    st.stop()

# Cargar el archivo CSV
datos = pd.read_csv(csv)

# Validar que tenga las columnas requeridas
colummnaRequerida = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
if not set(colummnaRequerida).issubset(datos.columns):
    st.error(f"El archivo debe contener las columnas: {', '.join(colummnaRequerida)}")
    st.stop()

# Combinar columnas de año y mes para la evolución temporal
datos["Año-Mes"] = datos["Año"].astype(str) + "-" + datos["Mes"].astype(str).str.zfill(2)

# Sidebar para elegir sucursal
sucursal = st.sidebar.selectbox(
    "Seleccionar Sucursal",
    options=["Todas las sucursales"] + sorted(datos["Sucursal"].unique())
)

# Filtrar datos por sucursal si corresponde
if sucursal != "Todas las sucursales":
    datos = datos[datos["Sucursal"] == sucursal]

# Título principal
st.title(f"Datos de Ventas - {sucursal}")

for producto in datos["Producto"].unique():
    diferenciaProducto = datos[datos["Producto"] == producto]

    # Calcular métricas de cada mes
    diferenciaProducto["Fecha"] = pd.to_datetime(diferenciaProducto["Año"].astype(str) + "-" + diferenciaProducto["Mes"].astype(str).str.zfill(2))
    diferenciaProducto = diferenciaProducto.sort_values(by="Fecha")
    
    # Calcular las métricas mes a mes
    diferenciaProducto["Precio Promedio"] = diferenciaProducto["Ingreso_total"] / diferenciaProducto["Unidades_vendidas"]
    diferenciaProducto["Margen Promedio"] = (diferenciaProducto["Ingreso_total"] - diferenciaProducto["Costo_total"]) / diferenciaProducto["Ingreso_total"]
    
    # Crear una nueva columna para las variaciones de precio, margen y unidades
    diferenciaProducto["Cambio Precio (%)"] = diferenciaProducto["Precio Promedio"].pct_change() * 100
    diferenciaProducto["Cambio Margen (%)"] = diferenciaProducto["Margen Promedio"].pct_change() * 100
    diferenciaProducto["Cambio Unidades (%)"] = diferenciaProducto["Unidades_vendidas"].pct_change() * 100

    # Mostrar métricas generales del producto
    st.subheader(producto)
    col1, col2= st.columns([1,3])
    
    # Mostramos las métricas promedio de todo el periodo
    precio_promedio = diferenciaProducto["Precio Promedio"].mean()
    margen_promedio = diferenciaProducto["Margen Promedio"].mean()
    unidadesVendas = diferenciaProducto["Unidades_vendidas"].sum()

    with st.container():
        with col1:
            col1.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{diferenciaProducto['Cambio Precio (%)'].iloc[-1]:.2f}%")
            col1.metric("Margen Promedio", f"{margen_promedio * 100:.2f}%", f"{diferenciaProducto['Cambio Margen (%)'].iloc[-1]:.2f}%")
            col1.metric("Unidades Vendidas", f"{unidadesVendas:,}", f"{diferenciaProducto['Cambio Unidades (%)'].iloc[-1]:.2f}%")

        with col2:
            # Gráfico de evolución de Unidades Vendidas
            fig, grafico = plt.subplots(figsize=(20, 8))  # Aumentamos el tamaño del gráfico para más espacio
            grafico.plot(diferenciaProducto["Año-Mes"], diferenciaProducto["Unidades_vendidas"], label="Unidades Vendidas", color="blue")
            
            # Línea de tendencia
            z = np.polyfit(range(len(diferenciaProducto)), diferenciaProducto["Unidades_vendidas"], 1)
            tendencia = np.poly1d(z)(range(len(diferenciaProducto)))
            grafico.plot(diferenciaProducto["Año-Mes"], tendencia, label="Tendencia", linestyle="--", color="red")
            
            # Mejoramos la visualización
            grafico.set_title(f"Evolución de Ventas - {producto}")
            grafico.set_xlabel("Año-Mes")
            grafico.set_ylabel("Unidades Vendidas")
                                        
            # Ajuste para que los ticks no se amontonen
            meses = diferenciaProducto["Año-Mes"].unique()
            grafico.set_xticks(meses[::3])  # Mostrar cada 3 meses
            grafico.set_xticklabels(meses[::3], rotation=45, ha="right")  # Rotamos y alineamos mejor los ticks
            
            grafico.legend()
            grafico.grid(True)
                                        
            # Crear un espacio grande al costado izquierdo del gráfico para las métricas
            fig.subplots_adjust(left=0.25)  # Mueve el gráfico para dar espacio a la izquierda
            st.pyplot(fig)  # Mostrar el gráfico
    st.markdown("<hr>", unsafe_allow_html=True)

