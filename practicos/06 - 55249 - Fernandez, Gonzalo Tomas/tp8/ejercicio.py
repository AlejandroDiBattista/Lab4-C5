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
        <a href="https://ventas-c7y7zul7tmul6f3jsuhmtj.streamlit.app/" target="_blank" style="text-decoration: none; color: red;">
            👉 Ver proyecto en la web
        </a>
        """, 
        unsafe_allow_html=True
    )



# Cargar archivo CSV
def cargar_datos():
    st.sidebar.title('Cargar archivo de datos')
    archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type="csv")
    return pd.read_csv(archivo_csv) if archivo_csv else None

# Calcular métricas agregadas por producto
def calcular_metricas(datos):
    datos['Precio_promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    datos['Ganancia'] = datos['Ingreso_total'] - datos['Costo_total']
    datos['Margen'] = (datos['Ganancia'] / datos['Ingreso_total']) * 100

    # Agregar métricas por producto
    return datos.groupby('Producto', sort=False).agg({
        'Precio_promedio': 'mean',
        'Margen': 'mean',
        'Unidades_vendidas': 'sum'
    }).reset_index()

# Calcular variaciones anuales por producto
def calcular_variaciones(datos, productos):
    variaciones = {}
    for producto in productos:
        datos_producto = datos[datos['Producto'] == producto]
        variaciones_producto = {}
        for metrica in ['Precio_promedio', 'Margen', 'Unidades_vendidas']:
            agrupado = datos_producto.groupby('Año')[metrica].sum() if metrica == 'Unidades_vendidas' else datos_producto.groupby('Año')[metrica].mean()
            variaciones_producto[metrica] = agrupado.pct_change().mean() * 100
        variaciones[producto] = variaciones_producto
    return variaciones

# Formatear valores para métricas
def formatear_metricas(valor, variacion=None):
    valor_formateado = f"${int(valor):,}".replace(",", ".") if isinstance(valor, (int, float)) else str(valor)
    variacion_formateada = f"{variacion:.2f}%" if variacion is not None else "N/A"
    return valor_formateado, variacion_formateada

# Generar gráfico de evolución de ventas con tendencia
def graficar_evolucion(datos, producto):
    datos_producto = datos[datos["Producto"] == producto].groupby(["Año", "Mes"], sort=False)["Unidades_vendidas"].sum().reset_index()
    datos_producto["Fecha"] = pd.to_datetime(datos_producto.rename(columns={"Año": "year", "Mes": "month"})[["year", "month"]].assign(day=1))
    
    x = np.arange(len(datos_producto))
    y = datos_producto["Unidades_vendidas"].values
    m, b = np.polyfit(x, y, 1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(datos_producto["Fecha"], y, label=producto)
    ax.plot(datos_producto["Fecha"], m * x + b, label="Tendencia", color="red", linestyle="--")

    ax.set_title(f"Evolución de Ventas Mensual")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(title='Producto')
    ax.grid(True)
    
    return fig

# Función principal
def main():
    datos = cargar_datos()
    if datos is None:
        mostrar_informacion_alumno()
        return

    # Seleccionar sucursal
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + datos["Sucursal"].unique().tolist())
    if sucursal_seleccionada != "Todas":
        datos = datos[datos["Sucursal"] == sucursal_seleccionada]

    metricas_por_producto = calcular_metricas(datos)
    variaciones = calcular_variaciones(datos, metricas_por_producto["Producto"])

    st.title(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

    for _, fila in metricas_por_producto.iterrows():
        producto = fila['Producto']
        precio_promedio, delta_precio = formatear_metricas(fila['Precio_promedio'], variaciones[producto]['Precio_promedio'])
        margen_promedio, delta_margen = formatear_metricas(fila['Margen'], variaciones[producto]['Margen'])
        unidades_vendidas, delta_unidades = formatear_metricas(fila['Unidades_vendidas'], variaciones[producto]['Unidades_vendidas'])

        with st.container(border=True):
            st.subheader(producto)
            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric("Precio Promedio", precio_promedio, delta_precio)
                st.metric("Margen Promedio", margen_promedio, delta_margen)
                st.metric("Unidades Vendidas", unidades_vendidas, delta_unidades)

            with col2:
                fig = graficar_evolucion(datos, producto)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
