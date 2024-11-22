import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL de la aplicación publicada
url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 80vh; text-align: center;">
            <div>
                <h2>Información del Alumno</h2>
                <p><strong>Legajo:</strong> 58866</p>
                <p><strong>Nombre:</strong> Velazquez, Tobias Alejo</p>
                <p><strong>Comisión:</strong> C5</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def cargar_datos():
    st.sidebar.header("📂 Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        return datos
    return None

def calcular_indicadores(datos):
    datos['Precio_Promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    datos['Margen_Promedio'] = (datos['Ingreso_total'] - datos['Costo_total']) / datos['Ingreso_total']
    return datos

def calcular_cambios(datos, producto):
    datos_producto = datos[datos['Producto'] == producto]
    precio_historico = datos_producto['Ingreso_total'].sum() / datos_producto['Unidades_vendidas'].sum()
    margen_historico = ((datos_producto['Ingreso_total'].sum() - datos_producto['Costo_total'].sum()) / 
                        datos_producto['Ingreso_total'].sum())
    unidades_historico = datos_producto['Unidades_vendidas'].sum()

    precio_actual = datos_producto['Precio_Promedio'].mean()
    margen_actual = datos_producto['Margen_Promedio'].mean()
    unidades_actual = datos_producto['Unidades_vendidas'].sum()

    cambio_precio = ((precio_actual - precio_historico) / precio_historico) * 100
    cambio_margen = ((margen_actual - margen_historico) / margen_historico) * 100
    cambio_unidades = ((unidades_actual - unidades_historico) / unidades_historico) * 100

    return cambio_precio, cambio_margen, cambio_unidades

def graficar_evolucion(datos, producto):
    datos_producto = datos[datos['Producto'] == producto]
    datos_producto['Año-Mes'] = datos_producto['Año'].astype(str) + '-' + datos_producto['Mes'].astype(str).str.zfill(2)
    datos_producto = datos_producto.groupby('Año-Mes').sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(datos_producto['Año-Mes'], datos_producto['Unidades_vendidas'], label=f'{producto}')
    z = np.polyfit(range(len(datos_producto)), datos_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(datos_producto['Año-Mes'], p(range(len(datos_producto))), label='Tendencia', linestyle='--', color='red')
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades vendidas')
    ax.legend()
    ax.set_xticks(range(0, len(datos_producto), max(1, len(datos_producto) // 10)))
    ax.set_xticklabels(datos_producto['Año-Mes'][::max(1, len(datos_producto) // 10)], rotation=45, ha='right')
    
    return fig

def mostrar_informacion(datos):
    sucursales = ["Todas"] + list(datos['Sucursal'].unique())
    sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)
    if sucursal != "Todas":
        datos = datos[datos['Sucursal'] == sucursal]
    
    st.header(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")
    productos = datos['Producto'].unique()
    
    # CSS para estilizar cada contenedor
    st.markdown(
        """
        <style>
            .info-container {
                border: 2px solid black;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
                background-color: #f9f9f9;
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    for producto in productos:
        # Contenedor estilizado para cada producto
        st.markdown(f'<div class="info-container">', unsafe_allow_html=True)

        # Incluir columnas y gráfico dentro del contenedor
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.markdown(f"### **{producto}**")
            producto_datos = datos[datos['Producto'] == producto]
            
            precio_promedio = producto_datos['Precio_Promedio'].mean()
            margen_promedio = producto_datos['Margen_Promedio'].mean() * 100
            unidades_vendidas = producto_datos['Unidades_vendidas'].sum()

            cambio_precio, cambio_margen, cambio_unidades = calcular_cambios(datos, producto)

            # Mostrar métricas con porcentajes
            st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{cambio_precio:+.2f}%", delta_color="inverse")
            st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{cambio_margen:+.2f}%", delta_color="inverse")
            st.metric("Unidades Vendidas", f"{unidades_vendidas:,.0f}", f"{cambio_unidades:+.2f}%", delta_color="inverse")

        with col2:
            fig = graficar_evolucion(datos, producto)
            st.pyplot(fig)

        # Cierre del contenedor estilizado
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    datos = cargar_datos()
    if datos is None:
        mostrar_informacion_alumno()
    else:
        datos = calcular_indicadores(datos)
        mostrar_informacion(datos)

if __name__ == "__main__":
    main()
