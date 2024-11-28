import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58758.streamlit.app/'

def mostrar_informacion_alumno():
        with st.container(border=True):
            st.markdown("#### **Legajo:** 58758")
            st.markdown("#### **Nombre:** Milagro Juarez")
            st.markdown("#### **Comisión:** C5")

def obtener_datos_archivo(archivo):
    try:
        df = pd.read_csv(archivo)
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str))
        df[['Ingreso_total', 'Unidades_vendidas', 'Costo_total']] = df[['Ingreso_total', 'Unidades_vendidas', 'Costo_total']].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=['Ingreso_total', 'Unidades_vendidas', 'Costo_total'], inplace=True)
        df['Precio Medio'] = df['Ingreso_total'] / df['Unidades_vendidas']
        df['Margen Neto'] = ((df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']) * 100
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

def calcular_variaciones(df):
    df_anual = df.groupby(['Producto', 'Año']).agg(
        Precio_Medio=('Precio Medio', 'mean'),
        Margen_Neto=('Margen Neto', 'mean'),
        Unidades_vendidas=('Unidades_vendidas', 'sum')
    ).reset_index()
    
    variaciones = df_anual.groupby('Producto').agg(
        cambio_precio=('Precio_Medio', lambda x: x.pct_change().mean() * 100),
        cambio_margen=('Margen_Neto', lambda x: x.pct_change().mean() * 100),
        cambio_unidades=('Unidades_vendidas', lambda x: x.pct_change().mean() * 100)
    ).reset_index()

    return df_anual, variaciones

def generar_grafico_ventas(df, producto):
    df_producto = df[df['Producto'] == producto].sort_values('Fecha')
    x = np.arange(len(df_producto))
    y = df_producto['Unidades_vendidas'].values
    tendencia = np.poly1d(np.polyfit(x, y, 1))(x)

    plt.figure(figsize=(10, 6))
    plt.plot(df_producto['Fecha'], y, label=producto, color='#3b87bb', linewidth=2)
    plt.plot(df_producto['Fecha'], tendencia, label='Tendencia', color='red', linestyle='--', linewidth=2)
    plt.xlabel("Año-Mes", fontsize=18)
    plt.ylabel("Unidades Vendidas", fontsize=12)
    plt.title(f"Evolución de Ventas Mensual - {producto}", fontsize=25, fontweight="bold")
    plt.legend(fontsize=10)
    
    y_min, y_max = int(np.floor(y.min() / 1000) * 1000), int(np.ceil(y.max() / 1000) * 1000)
    y_ticks = np.arange(y_min, y_max + 1000, 1000)
    plt.yticks(y_ticks, [f"{int(tick):,}" for tick in y_ticks], fontsize=10)

    x_ticks = pd.date_range(start=df_producto['Fecha'].min(), end=df_producto['Fecha'].max(), freq='YS')
    x_lines = pd.date_range(start=df_producto['Fecha'].min(), end=df_producto['Fecha'].max(), freq='MS')

    plt.xticks(x_ticks, [date.strftime("%Y") for date in x_ticks], rotation=0, fontsize=10)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    for date in x_lines:
        plt.axvline(date, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    st.pyplot(plt)

def ejecutar_app():
    st.title("Cargar archivo de datos")
    mostrar_informacion_alumno()

    archivo = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
    sucursal_elegida = st.sidebar.selectbox(
        "Elegir Sucursal",
        options=['Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
    )

    if archivo:
        df = obtener_datos_archivo(archivo)
        if df is not None:
            st.session_state.df = df  # Guardar los datos en el estado de sesión

            if sucursal_elegida != 'Todas':
                df = df[df['Sucursal'] == sucursal_elegida]

            st.header(f"Datos de {'Todas las Sucursales' if sucursal_elegida == 'Todas' else sucursal_elegida}")
            df_anual, variaciones = calcular_variaciones(df)

            for producto in df['Producto'].unique():
                df_producto = df[df['Producto'] == producto]
                
                with st.container():
                    st.markdown("---")
                st.markdown(f"### {producto}")

                col1, col2 = st.columns([0.25, 0.75])  # Ajustar proporciones de las columnas

                with col1:
                    precio_promedio = round(df_producto['Precio Medio'].mean(), 0)
                    margen_promedio = round(df_producto['Margen Neto'].mean(), 0)
                    unidades_vendidas = df_producto['Unidades_vendidas'].sum()

                    cambio_precio = variaciones.loc[variaciones['Producto'] == producto, 'cambio_precio'].values[0]
                    cambio_margen = variaciones.loc[variaciones['Producto'] == producto, 'cambio_margen'].values[0]
                    cambio_unidades = variaciones.loc[variaciones['Producto'] == producto, 'cambio_unidades'].values[0]

                    st.metric("Precio Promedio", f"${precio_promedio:,.0f}", f"{cambio_precio:+.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:,.0f}%", f"{cambio_margen:+.2f}%")
                    st.metric("Unidades Vendidas", f"{int(unidades_vendidas):,}", f"{cambio_unidades:+.2f}%")

                with col2:
                    generar_grafico_ventas(df_producto, producto)

if __name__ == "__main__":
    ejecutar_app()
