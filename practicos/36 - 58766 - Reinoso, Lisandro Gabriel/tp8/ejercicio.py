import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58766.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58766')
        st.markdown('**Nombre:** Reinoso Lisandro Gabriel')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()

st.sidebar.title("Cargar archivos de datos")
archivo = st.sidebar.file_uploader("Subí el archivo CSV", type=["csv"])

if archivo:
    sucursales = pd.read_csv(archivo)
    
    lista_sucursales = sucursales["Sucursal"].drop_duplicates().to_list()
    lista_sucursales.insert(0, "Todas")
    opcion_elegida = st.sidebar.selectbox("Seleccionar sucursal", lista_sucursales)

    if opcion_elegida == "Todas":
        st.title("Datos de todas las sucursales")
        Por_sucursal_o_todas = sucursales
    else:
        st.title("Datos de " + opcion_elegida)
        Por_sucursal_o_todas = sucursales[sucursales["Sucursal"] == opcion_elegida]

    productos_agrupados = Por_sucursal_o_todas.groupby("Producto")

    for producto, dato_producto_agrupado in productos_agrupados:
        
        with st.container(border=True):
            dato_producto_agrupado["Año_x_Mes"] = pd.to_datetime(dato_producto_agrupado["Año"].astype(str) + "-" + dato_producto_agrupado["Mes"].astype(str).str.zfill(2))
            rango_tiempo = pd.DataFrame({"Año_x_Mes": pd.date_range(start=dato_producto_agrupado["Año_x_Mes"].min(),end=dato_producto_agrupado["Año_x_Mes"].max(),freq="MS")})
            dato_producto_agrupado = rango_tiempo.merge(dato_producto_agrupado, on="Año_x_Mes", how="left").fillna(0)

            dato_producto_agrupado["Unidades_vendidas_suavizadas"] = dato_producto_agrupado["Unidades_vendidas"].rolling(window=6, min_periods=1).mean()

            dato_producto_agrupado["precio_por_producto"] = dato_producto_agrupado["Ingreso_total"] / dato_producto_agrupado["Unidades_vendidas"]
            precio_promedio = dato_producto_agrupado["precio_por_producto"].mean()

            dato_producto_agrupado["Margen"] = ((dato_producto_agrupado["Ingreso_total"] - dato_producto_agrupado["Costo_total"]) / dato_producto_agrupado["Ingreso_total"]) * 100
            margen_promedio = dato_producto_agrupado["Margen"].mean()
            total_unidades_vendidas = dato_producto_agrupado["Unidades_vendidas"].sum()

            precio_promedio_anual = dato_producto_agrupado.groupby("Año")["precio_por_producto"].mean()
            variacion_precio = precio_promedio_anual.pct_change().mean() * 100

            margen_promedio_anual = dato_producto_agrupado.groupby("Año")["Margen"].mean()
            variacion_margen = margen_promedio_anual.pct_change().mean() * 100

            unidades_vendidas_anuales = dato_producto_agrupado.groupby("Año")["Unidades_vendidas"].sum()
            variacion_unidades = unidades_vendidas_anuales.pct_change().mean() * 100

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader(f"{producto}")
                st.metric("Precio Promedio", f"${precio_promedio:,.0f}", f"{variacion_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio.round():.2f}%", f"{variacion_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{total_unidades_vendidas:,.0f}", f"{variacion_unidades:.2f}%")

            with col2:
                fig, ax = plt.subplots(figsize=(14, 8))  
                ax.plot(dato_producto_agrupado["Año_x_Mes"], dato_producto_agrupado["Unidades_vendidas_suavizadas"], label=f"{producto}", color="blue", linewidth=2)

                x_numeric = np.arange(len(dato_producto_agrupado))
                tendencia = np.polyfit(x_numeric, dato_producto_agrupado["Unidades_vendidas_suavizadas"], 1)
                trend = np.poly1d(tendencia)
                ax.plot(dato_producto_agrupado["Año_x_Mes"], trend(x_numeric), label="Tendencia", color="red", linestyle="--", linewidth=2)
                
                ax.set_ylim(0, None) 
                ax.set_xticks(pd.date_range(start=dato_producto_agrupado["Año_x_Mes"].min(), end=dato_producto_agrupado["Año_x_Mes"].max(),freq="12MS"))
                ax.set_xticklabels(pd.date_range(start=dato_producto_agrupado["Año_x_Mes"].min(),end=dato_producto_agrupado["Año_x_Mes"].max(),freq="12MS").strftime("%Y-%m"), rotation=45, ha="right", fontsize=12)
                ax.tick_params(axis='both', labelsize=14)
                ax.set_xlabel("Año - Mes", fontsize=18)
                ax.set_ylabel("Unidades Vendidas", fontsize=18)
                ax.set_title("Evolución de Ventas Mensual", fontsize=20)
                ax.legend(fontsize=14)  
                ax.grid(alpha=0.4)
                
                st.pyplot(fig)