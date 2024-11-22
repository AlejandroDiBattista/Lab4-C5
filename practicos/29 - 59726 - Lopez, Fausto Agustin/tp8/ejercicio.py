import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#URL=https://2doparcial-faustolopez.streamlit.app/

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58726')
        st.markdown('**Nombre:** Fausto López')
        st.markdown('**Comisión:** C5')

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
    ax.set_xticks(x_vals[::max(len(x_vals) // 5, 0)])
    ax.set_xticklabels(resumen_ventas['Periodo'][::max(len(x_vals) // 5, 0)], rotation=45, ha='right')
    ax.set_ylabel("Unidades Vendidas")
    ax.set_ylim(0, None) 
    ax.grid(alpha=0.5, linestyle=":")
    ax.legend()
    
    return fig


mostrar_informacion_alumno()

st.sidebar.header("Cargar archivos de datos")
archivo = st.sidebar.file_uploader("Subir archivo CSV", type=['csv'])

columnas_requeridas = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]

if archivo is not None:
    df = pd.read_csv(archivo)

    if not all(col in df.columns for col in columnas_requeridas):
        st.error("El archivo no tiene las columnas requeridas")
    else:
        st.sidebar.subheader("Seleccionar Sucursal")
        opciones_sucursales = ["Todas"] + list(df['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccione una sucursal", opciones_sucursales)

        if sucursal_seleccionada == "Todas":
            df_filtrado = df
        else:
            df_filtrado = df[df['Sucursal'] == sucursal_seleccionada]

        st.title(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

        productos = df_filtrado["Producto"].unique()

        for producto in productos:
            df_producto = df_filtrado[df_filtrado["Producto"] == producto]

            df_producto['Precio_promedio'] = df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']
            precio_promedio = df_producto['Precio_promedio'].mean()

            df_producto['Ganancia'] = df_producto['Ingreso_total'] - df_producto['Costo_total']
            df_producto['Margen'] = (df_producto['Ganancia'] / df_producto['Ingreso_total']) * 100
            margen_promedio = df_producto['Margen'].mean()

            unidades_vendidas = df_producto['Unidades_vendidas'].sum()

            precio_promedio_anual = df_producto.groupby('Año')['Precio_promedio'].mean()
            variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

            margen_promedio_anual = df_producto.groupby('Año')['Margen'].mean()
            variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

            unidades_por_año = df_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

            col1, col2 = st.columns([0.25, 0.75])

            with col1:
                st.subheader(producto)
                st.metric("Precio Promedio", f"${precio_promedio:,.2f}".replace(",", "."), delta=f"{variacion_precio_promedio_anual:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.2f}%", delta=f"{variacion_margen_promedio_anual:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_anual_unidades:.2f}%")

            with col2:
                fig = generar_grafico_evolucion(df_producto, producto)
                st.pyplot(fig)
