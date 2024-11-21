import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mostrar información del alumno
def mostrar_informacion_alumno():
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container(border=True):
        st.markdown('**Legajo:** 55.249')
        st.markdown('**Nombre:** Gonzalo Tomas')
        st.markdown('**Comisión:** C5')
    st.markdown(
            """
            <a href="https://ventas-c7y7zul7tmul6f3jsuhmtj.streamlit.app/" target="_blank" style="text-decoration: none; color: blue;">
                👉 Ver proyecto en la web
            </a>
            """, 
            unsafe_allow_html=True
        )

# Cargar y procesar archivo CSV
def cargar_datos():
    st.sidebar.title('Cargar archivo de datos')
    archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
    return pd.read_csv(archivo_csv) if archivo_csv else None

# Calcular métricas por producto y porcentaje de cambio
def calcular_metricas(datos, sucursal_seleccionada):
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]

    resumen = datos.groupby("Producto").agg({
        "Ingreso_total": "sum",
        "Costo_total": "sum",
        "Unidades_vendidas": "sum"
    }).reset_index()

    resumen["Precio_promedio"] = resumen["Ingreso_total"] / resumen["Unidades_vendidas"]
    resumen["Margen_promedio"] = (resumen["Ingreso_total"] - resumen["Costo_total"]) / resumen["Ingreso_total"]
    resumen["Cambio_precio"] = resumen["Precio_promedio"].pct_change().fillna(0) * 100
    resumen["Cambio_margen"] = resumen["Margen_promedio"].pct_change().fillna(0) * 100
    resumen["Cambio_unidades"] = resumen["Unidades_vendidas"].pct_change().fillna(0) * 100
    
    return resumen

# Gráfico de evolución de ventas mensuales con tendencia y métricas adicionales
def graficar_evolucion(datos, producto):
    datos_producto = datos[datos["Producto"] == producto].groupby(["Año", "Mes"])["Unidades_vendidas"].sum().reset_index()
    datos_producto["Fecha"] = pd.to_datetime(datos_producto.rename(columns={"Año": "year", "Mes": "month"})[["year", "month"]].assign(day=1))
    
    x = np.arange(len(datos_producto))
    y = datos_producto["Unidades_vendidas"].values
    m, b = np.polyfit(x, y, 1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(datos_producto["Fecha"], y, label=producto)
    ax.plot(datos_producto["Fecha"], m * x + b, label="Tendencia", color="red", linestyle="--")

    # Configuración de ticks anuales y mensuales
    fechas_anos = datos_producto["Fecha"].dt.to_period("Y").drop_duplicates().dt.to_timestamp()
    ax.set_xticks(fechas_anos)
    ax.set_xticklabels(fechas_anos.dt.strftime("%Y"))
    ax.set_xticks(pd.date_range(datos_producto["Fecha"].min(), datos_producto["Fecha"].max(), freq="M"), minor=True)

    ax.tick_params(axis="x", which="both", length=4.4)
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.set_title("Evolución de Ventas Mensual")
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(title="Producto")
    
    return fig

# Función principal
def main():
    datos = cargar_datos()
    if datos is None:
        mostrar_informacion_alumno()
    else:
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + datos["Sucursal"].unique().tolist())
        resumen = calcular_metricas(datos, sucursal_seleccionada)
        
        st.title(f"Datos de {sucursal_seleccionada if sucursal_seleccionada != 'Todas' else 'Todas las Sucursales'}")
        for _, fila in resumen.iterrows():
            with st.container(border=True):
                st.subheader(fila["Producto"])
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric("Precio Promedio", f"${fila['Precio_promedio']:.2f}", f"{fila['Cambio_precio']:.2f}%")
                    st.metric("Margen Promedio", f"{fila['Margen_promedio']:.0%}", f"{fila['Cambio_margen']:.2f}%")
                    st.metric("Unidades Vendidas", f"{int(fila['Unidades_vendidas']):,}", f"{fila['Cambio_unidades']:.2f}%")
                
                with col2:
                    fig = graficar_evolucion(datos, fila["Producto"])
                    st.pyplot(fig)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
