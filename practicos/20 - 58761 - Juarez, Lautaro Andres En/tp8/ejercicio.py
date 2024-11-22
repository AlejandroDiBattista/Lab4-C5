import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#link de streamlit
#     https://lautarojuarezparcial.streamlit.app/

def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58761')
        st.markdown('**Nombre:** Lautaro Juarez')
        st.markdown('**Comisión:** C5')

def cargar_datos(file):
    try:
        df = pd.read_csv(file)
        df['Año-Mes'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str))
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

def calcular_estadisticas(df, sucursal_seleccionada):
    if sucursal_seleccionada != 'Todas':
        df = df[df['Sucursal'] == sucursal_seleccionada]
    
    for col in ['Ingreso_total', 'Unidades_vendidas', 'Costo_total']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['Ingreso_total', 'Unidades_vendidas', 'Costo_total'])
    
    df['Precio Promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen'] = ((df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']) * 100

    return df

def calcular_variaciones(df):
    variaciones = {}
    for producto in df['Producto'].unique():
        datos_producto = df[df['Producto'] == producto]
        precio_promedio_anual = datos_producto.groupby('Año')['Precio Promedio'].mean()
        margen_promedio_anual = datos_producto.groupby('Año')['Margen'].mean()
        unidades_vendidas_anual = datos_producto.groupby('Año')['Unidades_vendidas'].sum()

        variaciones[producto] = {
            "variacion_precio_promedio": precio_promedio_anual.pct_change().mean() * 100,
            "variacion_margen_promedio": margen_promedio_anual.pct_change().mean() * 100,
            "variacion_unidades_vendidas": unidades_vendidas_anual.pct_change().mean() * 100
        }
    return variaciones

def graficar_desarrollo(df, producto):
    df_producto = df[df['Producto'] == producto].sort_values('Año-Mes')
    x = np.arange(len(df_producto))
    y = df_producto['Unidades_vendidas'].values

    tendencia = np.poly1d(np.polyfit(x, y, 1))(x)
    
    plt.figure(figsize=(10, 6))  
    plt.plot(df_producto['Año-Mes'], y, label=producto, color='#3b87bb', linewidth=2)  # Línea principal
    plt.plot(df_producto['Año-Mes'], tendencia, label='Tendencia', color='red', linestyle='--', linewidth=2)  # Línea de tendencia
    
    plt.xlabel("Año-Mes", fontsize=12)
    plt.ylabel("Unidades Vendidas", fontsize=12)
    plt.title("Evolución de Ventas Mensual", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    
    y_min = int(np.floor(y.min() / 1000) * 1000)
    y_max = int(np.ceil(y.max() / 1000) * 1000)   
    y_ticks = np.arange(y_min, y_max + 1000, 1000)  
    plt.yticks(y_ticks, [f"{int(tick):,}" for tick in y_ticks], fontsize=10)  
    
    x_ticks = pd.date_range(start=df_producto['Año-Mes'].min(), 
                            end=df_producto['Año-Mes'].max(), 
                            freq='YS')  
    x_lines = pd.date_range(start=df_producto['Año-Mes'].min(), 
                            end=df_producto['Año-Mes'].max(), 
                            freq='MS')  
    
    plt.xticks(x_ticks, [date.strftime("%Y") for date in x_ticks], rotation=0, fontsize=10)

    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    for date in x_lines:
        plt.axvline(date, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Líneas verticales mensuales
    
    plt.tight_layout()  

    st.pyplot(plt)  

def main():
    st.title("Análisis de Ventas")
    mostrar_informacion_alumno()
    
    file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    sucursal_seleccionada = st.sidebar.selectbox(
        "Seleccionar Sucursal",
        options=['Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
    )
    
    if file:
        df = cargar_datos(file)
        if df is not None:
            st.header(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
            
            df = calcular_estadisticas(df, sucursal_seleccionada)
            variaciones = calcular_variaciones(df)
            
            for producto in df['Producto'].unique():
                datos_producto = df[df['Producto'] == producto]

                st.markdown(f"### {producto}")

                col1, col2 = st.columns([1, 2])  

                with col1:
                    precio_promedio = datos_producto['Precio Promedio'].mean()
                    margen_promedio = datos_producto['Margen'].mean()
                    unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

                    variacion_precio = variaciones[producto]["variacion_precio_promedio"]
                    variacion_margen = variaciones[producto]["variacion_margen_promedio"]
                    variacion_unidades = variaciones[producto]["variacion_unidades_vendidas"]

                    st.metric(
                        "Precio Promedio",
                        f"${precio_promedio:.2f}",
                        f"{variacion_precio:+.2f}%"
                    )
                    st.metric(
                        "Margen Promedio",
                        f"{margen_promedio:.2f}%",
                        f"{variacion_margen:+.2f}%"
                    )
                    st.metric(
                        "Unidades Vendidas",
                        f"{int(unidades_vendidas):,}",
                        f"{variacion_unidades:+.2f}%"
                    )

                with col2:
                    graficar_desarrollo(datos_producto, producto)

if __name__ == "__main__":
    main()
