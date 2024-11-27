import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://tp8-lab-xvbgxpfxxdt45ydttjuhgj.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.753')
        st.markdown('**Nombre:** Lautaro Rivadeneira')
        st.markdown('**Comisión:** C5')

def cargar_datos():
    st.sidebar.title("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        return datos
    return None

def calcular_estadisticas(datos, sucursal):
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]

    datos = datos.dropna(subset=["Unidades_vendidas", "Ingreso_total", "Costo_total", "Año", "Mes"])
    datos = datos[(datos["Unidades_vendidas"] > 0) & (datos["Ingreso_total"] > 0)]
    datos["Año"] = datos["Año"].astype(int)
    datos["Mes"] = datos["Mes"].astype(int)
    
    datos["Precio_promedio"] = datos["Ingreso_total"] / datos["Unidades_vendidas"]
    datos["Margen_promedio"] = (datos["Ingreso_total"] - datos["Costo_total"]) / datos["Ingreso_total"]
    
    estadisticas_list = []
    
    for producto, group_producto in datos.groupby('Producto', sort=False):
        promedio_precio_anual = group_producto.groupby('Año')['Precio_promedio'].mean()
        promedio_margen_anual = group_producto.groupby('Año')['Margen_promedio'].mean()
        unidades_agrupadas_anual = group_producto.groupby('Año')['Unidades_vendidas'].sum()
        
        variacion_precio_anual = promedio_precio_anual.pct_change().mean() * 100
        variacion_margen_anual = promedio_margen_anual.pct_change().mean() * 100
        variacion_unidades_anual = unidades_agrupadas_anual.pct_change().mean() * 100
        
        variacion_precio_anual = variacion_precio_anual if not np.isnan(variacion_precio_anual) else 0
        variacion_margen_anual = variacion_margen_anual if not np.isnan(variacion_margen_anual) else 0
        variacion_unidades_anual = variacion_unidades_anual if not np.isnan(variacion_unidades_anual) else 0
        
        total_unidades_vendidas = group_producto['Unidades_vendidas'].sum()
        promedio_precio = group_producto['Precio_promedio'].mean()
        promedio_margen = group_producto['Margen_promedio'].mean()
        
        estadisticas_list.append({
            'Producto': producto,
            'Unidades_vendidas': total_unidades_vendidas,
            'Precio_promedio': promedio_precio,
            'Margen_promedio': promedio_margen,
            'Cambio_Unidades': variacion_unidades_anual,
            'Cambio_Precio': variacion_precio_anual,
            'Cambio_Margen': variacion_margen_anual
        })
    
    estadisticas = pd.DataFrame(estadisticas_list)
    estadisticas.set_index('Producto', inplace=True)
    return estadisticas

def mostrar_metricas(columna, titulo, valor, cambio):
    if "Precio" in titulo:
        valor_formateado = f"${valor:,.0f}"
    elif "Unidades" in titulo:
        valor_formateado = f"{valor:,.0f}"
    elif "Margen" in titulo:
        valor_formateado = f"{valor*100:.0f}%"
    else:
        valor_formateado = f"{valor}"
    
    columna.metric(
        titulo,
        valor_formateado,
        f"{cambio:+.2f}%"
    )

def graficar_evolucion(datos, producto):
    datos_producto = datos[datos["Producto"] == producto].copy()

    datos_producto["Año"] = pd.to_numeric(datos_producto["Año"], errors="coerce")
    datos_producto["Mes"] = pd.to_numeric(datos_producto["Mes"], errors="coerce")
    
    datos_producto = datos_producto.dropna(subset=["Año", "Mes"])

    datos_producto["Fecha"] = pd.to_datetime({
        "year": datos_producto["Año"].astype(int),
        "month": datos_producto["Mes"].astype(int),
        "day": 1
    }, errors="coerce")

    datos_producto = datos_producto.dropna(subset=["Fecha"])

    datos_producto.sort_values("Fecha", inplace=True)

    monthly_sales = datos_producto.groupby("Fecha").agg({
        "Unidades_vendidas": "sum"
    }).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(
        monthly_sales["Fecha"], monthly_sales["Unidades_vendidas"],
        marker="o", linestyle="-", label=producto
    )
    z = np.polyfit(range(len(monthly_sales)), monthly_sales["Unidades_vendidas"], 1)
    p = np.poly1d(z)
    plt.plot(monthly_sales["Fecha"], p(range(len(monthly_sales))), linestyle="--", color="red", label="Tendencia")
    plt.title(f"Evolución de Ventas Mensual - {producto}")
    plt.xlabel("Fecha (Año-Mes)")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def main():
    datos = cargar_datos()
    
    if datos is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
    else:
        sucursal = st.sidebar.selectbox(
            "Seleccionar Sucursal", ["Todas"] + sorted(datos["Sucursal"].unique())
        )
        
        if sucursal == "Todas":
            st.title("Datos de Todas las Sucursales")
        else:
            st.title(f"Datos de {sucursal}")
        
        estadisticas = calcular_estadisticas(datos, sucursal)
        
        for producto, row in estadisticas.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader(producto)
                    mostrar_metricas(col1, "Precio Promedio", row["Precio_promedio"], row["Cambio_Precio"])
                    mostrar_metricas(col1, "Margen Promedio", row["Margen_promedio"], row["Cambio_Margen"])
                    mostrar_metricas(col1, "Unidades Vendidas", row["Unidades_vendidas"], row["Cambio_Unidades"])
                with col2:
                    graficar_evolucion(datos, producto)
            st.markdown(" ")

if __name__ == "__main__":
    main()
