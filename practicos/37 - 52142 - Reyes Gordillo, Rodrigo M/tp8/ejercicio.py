import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Mostrar información del alumno
def mostrar_informacion_alumno():
    with st.container():
        st.markdown("**Legajo:** 52142")
        st.markdown("**Nombre:** Reyes Gordillo Rodrigo Maximiliano")
        st.markdown("**Comisión:** C5")

# Configuración de la página
st.set_page_config(page_title="Análisis de Ventas", layout="wide")

# Carga de datos con caché
@st.cache_data
def cargar_datos(archivo):
    return pd.read_csv(archivo)

# Generar gráfico de evolución
def generar_grafico_evolucion(datos_producto, nombre_producto):
    resumen_ventas = datos_producto.pivot_table(
        index=['Año', 'Mes'], values='Unidades_vendidas', aggfunc='sum'
    ).reset_index()

    resumen_ventas['Periodo'] = resumen_ventas['Año'].astype(str) + "-" + resumen_ventas['Mes'].astype(str).str.zfill(2)

    x_vals = range(len(resumen_ventas))
    y_vals = resumen_ventas['Unidades_vendidas']

    coef_tendencia = np.polyfit(x_vals, y_vals, deg=1)
    tendencia_lineal = np.poly1d(coef_tendencia)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals, y_vals, label=f"Ventas: {nombre_producto}", color="blue")
    ax.plot(x_vals, tendencia_lineal(x_vals), linestyle="--", color="red", label="Tendencia")
    
    ax.set_title("Evolución de Ventas Mensuales")
    ax.set_xlabel("Año-Mes")
    ax.set_xticks(x_vals[::max(len(x_vals) // 5, 1)])
    ax.set_xticklabels(resumen_ventas['Periodo'][::max(len(x_vals) // 5, 1)], rotation=45, ha='right')
    ax.set_ylabel("Unidades Vendidas")
    ax.set_ylim(0, None) 
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    
    return fig

# Configuración de la barra lateral
st.sidebar.header("Cargar archivo de datos")
archivo_cargado = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

columnas_esperadas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]

if archivo_cargado:
    datos = cargar_datos(archivo_cargado)

    if not all(col in datos.columns for col in columnas_esperadas):
        st.error("El archivo CSV debe contener las columnas: " + ", ".join(columnas_esperadas))
    else:
        # Filtro por sucursal
        sucursales = ["Todas"] + list(datos['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            datos = datos[datos['Sucursal'] == sucursal_seleccionada]

        st.title(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

        productos = datos['Producto'].unique()

        for producto in productos: 
            datos_producto = datos[datos["Producto"] == producto]

            # Agregar columnas calculadas al DataFrame
            datos_producto['Precio_promedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            datos_producto['Ganancia'] = datos_producto['Ingreso_total'] - datos_producto['Costo_total']
            datos_producto['Margen'] = (datos_producto['Ganancia'] / datos_producto['Ingreso_total']) * 100

            # Calcular promedios
            precio_promedio = round(datos_producto['Precio_promedio'].mean())  # Redondeo a entero
            margen_promedio = round(datos_producto['Margen'].mean())  # Redondeo a entero
            unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

            # Calcular tendencias anuales
            precio_promedio_anual = datos_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = round(precio_promedio_anual.pct_change().mean() * 100, 2)

            margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = round(margen_promedio_anual.pct_change().mean() * 100, 2)

            unidades_por_año = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = round(unidades_por_año.pct_change().mean() * 100, 2)

            # Formatear valores
            unidades_vendidas_redondeadas = f"{unidades_vendidas:,.0f}"  # Separador de miles

            # Mostrar métricas
            col1, col2 = st.columns([0.25, 0.75])
            with col1:
                st.subheader(producto)
                st.metric(
                    label="Precio Promedio",
                    value=f"${precio_promedio:,.0f}",  # Redondeado a entero
                    delta=f"{variacion_precio_promedio_anual:.2f}%",
                    delta_color="normal"
                )
                st.metric(
                    label="Margen Promedio",
                    value=f"{margen_promedio:.0f}%",  # Redondeado a entero
                    delta=f"{variacion_margen_promedio_anual:.2f}%",
                    delta_color="normal"
                )
                st.metric(
                    label="Unidades Vendidas",
                    value=unidades_vendidas_redondeadas,
                    delta=f"{variacion_anual_unidades:.2f}%",
                    delta_color="normal"
                )
            with col2:
                fig = generar_grafico_evolucion(datos_producto, producto)
                st.pyplot(fig)

else:
    mostrar_informacion_alumno()
