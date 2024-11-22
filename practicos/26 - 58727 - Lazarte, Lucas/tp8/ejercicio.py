import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://lazartelucas.streamlit.app'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.727')
        st.markdown('**Nombre:** Lazarte Lucas')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()

# Título de la aplicación
st.title("Datos de Todas las Sucursales")

# Diseño para la parte izquierda (cargar archivo y seleccionar sucursal)
st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

# Selección de sucursal
st.sidebar.subheader("Seleccionar Sucursal")
sucursales = ['Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

# Función para calcular el cambio porcentual
def calcular_cambio_porcentual(primero, ultimo):
    # Evitar división por cero o valores nulos
    if pd.isna(primero) or pd.isna(ultimo) or primero == 0:
        return 0
    # Ajustar el cálculo para evitar cambios porcentuales desproporcionados
    return (ultimo - primero) / abs(primero) * 100

# Función que calcula las métricas (Precio Promedio y Margen) y las agrega al DataFrame
def calcular_metricas(df):
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = ((df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']) * 100
    return df

# Si se sube un archivo
if uploaded_file is not None:
    try:
        # Leer los datos del CSV
        df = pd.read_csv(uploaded_file)

        # Verificar que las columnas necesarias existen
        required_columns = ['Ingreso_total', 'Unidades_vendidas', 'Costo_total', 'Producto', 'Sucursal', 'Año', 'Mes']
        for column in required_columns:
            if column not in df.columns:
                st.error(f"Falta la columna: {column}")
                st.stop()

        # Calcular las métricas (Precio Promedio y Margen Promedio)
        df = calcular_metricas(df)

        # Sumar las Unidades vendidas por Producto
        df_productos = df.groupby('Producto').agg(
            Precio_promedio=('Precio_promedio', 'mean'),
            Margen_promedio=('Margen_promedio', 'mean'),
            Unidades_vendidas=('Unidades_vendidas', 'sum')
        ).reset_index()

        # Filtro por sucursal
        if sucursal_seleccionada != 'Todas':
            df = df[df['Sucursal'] == sucursal_seleccionada]

        # Mostrar datos y gráficos por cada producto
        for producto_seleccionado in df['Producto'].unique():
            st.subheader(f"**{producto_seleccionado}**")
            df_producto = df[df['Producto'] == producto_seleccionado]

            # Crear la fecha tomando el primer día de cada mes
            df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str) + '-01')

            # Ordenar por fecha
            df_producto = df_producto.sort_values('Fecha')

            # Disposición de las columnas para mostrar los gráficos y estadísticas
            col1, col2 = st.columns([3, 4])  # 3 partes para las estadísticas y 4 partes para el gráfico

            with col1:
                # Mostrar las estadísticas de cada producto con mayor tamaño y énfasis
                precio_promedio = df_productos[df_productos['Producto'] == producto_seleccionado]['Precio_promedio'].values[0]
                margen_promedio = df_productos[df_productos['Producto'] == producto_seleccionado]['Margen_promedio'].values[0]
                unidades_vendidas = df_productos[df_productos['Producto'] == producto_seleccionado]['Unidades_vendidas'].values[0]

                # Calcular los cambios porcentuales
                cambio_precio = calcular_cambio_porcentual(df_producto['Precio_promedio'].iloc[0], df_producto['Precio_promedio'].iloc[-1])
                cambio_margen = calcular_cambio_porcentual(df_producto['Margen_promedio'].iloc[0], df_producto['Margen_promedio'].iloc[-1])
                cambio_unidades = calcular_cambio_porcentual(df_producto['Unidades_vendidas'].iloc[0], df_producto['Unidades_vendidas'].iloc[-1])

                # Usar st.metric con delta_color en 'normal' para los valores positivos y 'inverse' para los negativos
                st.metric(label="Precio Promedio", value=f"${precio_promedio:,.2f}",
                          delta=f"{cambio_precio:.2f}%", delta_color="normal" if cambio_precio >= 0 else "inverse")

                st.metric(label="Margen Promedio", value=f"{margen_promedio:.2f}%",
                          delta=f"{cambio_margen:.2f}%", delta_color="normal" if cambio_margen >= 0 else "inverse")

                st.metric(label="Unidades Vendidas", value=f"{unidades_vendidas:,.0f}",
                          delta=f"{cambio_unidades:.2f}%", delta_color="normal" if cambio_unidades >= 0 else "inverse")

                # CSS personalizado para asegurar que los valores positivos sean verdes y negativos rojos
                st.markdown("""
                    <style>
                        .stMetric .stMetricDelta {
                            color: green;
                        }
                        .stMetric .stMetricDelta.inverse {
                            color: red;
                        }
                    </style>
                """, unsafe_allow_html=True)

            with col2:
                # Graficar evolución de ventas
                plt.figure(figsize=(10, 6))
                plt.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], label=producto_seleccionado)

                # Ajustar una línea de tendencia utilizando polinomio de grado 1 (recta)
                z = np.polyfit(df_producto['Fecha'].map(pd.Timestamp.toordinal), df_producto['Unidades_vendidas'], 1)
                p = np.poly1d(z)
                plt.plot(df_producto['Fecha'], p(df_producto['Fecha'].map(pd.Timestamp.toordinal)), color='r', linestyle='--', label='Tendencia')

                plt.title(f'Evolución de Ventas Mensual - {producto_seleccionado}', fontsize=16)
                plt.xlabel('Año-Mes', fontsize=14)
                plt.ylabel('Unidades Vendidas', fontsize=14)
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(plt)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

else:
    st.warning("Cargue un archivo CSV para comenzar.")
