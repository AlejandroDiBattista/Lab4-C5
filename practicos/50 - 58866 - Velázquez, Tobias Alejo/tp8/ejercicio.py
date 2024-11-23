import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58866.streamlit.app/'

def mostrar_info_usuario():
    """
    Muestra la información del usuario al principio si no hay datos cargados.
    """
    with st.container(border=True):
        st.markdown('**Legajo:** 58866')
        st.markdown('**Nombre:** Velazquez, Tobias Alejo.')
        st.markdown('**Comisión:** C5')

def cargar_archivo():
    """
    Carga los datos desde un archivo CSV.
    """
    st.sidebar.header("Subir datos")
    archivo_csv = st.sidebar.file_uploader("Carga un archivo CSV", type=["csv"])
    if archivo_csv is not None:
        try:
            df = pd.read_csv(archivo_csv)
            df.columns = ['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades', 'Ingresos', 'Costos']
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    return None

def calcular_metrica(datos_df):
    """
    Calcula métricas como el precio promedio y el margen promedio.
    """
    datos_df['Promedio_Precio'] = datos_df['Ingresos'] / datos_df['Unidades']
    datos_df['Promedio_Margen'] = (datos_df['Ingresos'] - datos_df['Costos']) / datos_df['Ingresos']
    return datos_df

def calcular_variaciones(datos_df):
    """
    Calcula las variaciones porcentuales de precio, margen y unidades.
    """
    datos_df['Delta_Precio'] = datos_df['Promedio_Precio'].pct_change() * 100
    datos_df['Delta_Margen'] = datos_df['Promedio_Margen'].pct_change() * 100
    datos_df['Delta_Unidades'] = datos_df['Unidades'].pct_change() * 100
    return datos_df

def generar_grafico(datos_filtrados, producto_nombre):
    """
    Genera un gráfico de la evolución de ventas para un producto.
    """
    datos_prod = datos_filtrados[datos_filtrados['Producto'] == producto_nombre]
    datos_prod['Fecha'] = pd.to_datetime(
        datos_prod['Año'].astype(str) + '-' + datos_prod['Mes'].astype(str).str.zfill(2)
    )
    datos_prod = datos_prod.groupby('Fecha').sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(datos_prod['Fecha'], datos_prod['Unidades'], label=f'{producto_nombre}')
    coef = np.polyfit(range(len(datos_prod)), datos_prod['Unidades'], 1)
    polinomio = np.poly1d(coef)
    ax.plot(datos_prod['Fecha'], polinomio(range(len(datos_prod))), label='Tendencia', linestyle='--', color='red')

    ax.set_title('Evolución de Ventas Mensual', fontsize=14, fontweight='bold')
    ax.set_xlabel('Año-Mes', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)
    ax.legend()

    # Limitar el rango del eje X al rango real de las fechas
    ax.set_xlim(datos_prod['Fecha'].min(), datos_prod['Fecha'].max())

    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax.grid(visible=True, which='major', color='black', linestyle='-', linewidth=0.75)
    ax.grid(visible=True, which='minor', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.tight_layout()
    return fig

def mostrar_datos(data):
    """
    Muestra la información de las sucursales y productos.
    """
    opciones_sucursal = ["Todas"] + list(data['Sucursal'].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Elige una Sucursal", opciones_sucursal)
    if sucursal_seleccionada != "Todas":
        data = data[data['Sucursal'] == sucursal_seleccionada]

    st.header(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")
    lista_productos = data['Producto'].unique()

    for prod in lista_productos:
        data_producto = data[data['Producto'] == prod]
        precio_medio = data_producto['Promedio_Precio'].mean()
        margen_medio = data_producto['Promedio_Margen'].mean() * 100
        total_unidades = data_producto['Unidades'].sum()

        # Cálculo de las variaciones porcentuales
        variaciones = data_producto[['Delta_Precio', 'Delta_Margen', 'Delta_Unidades']].iloc[-1]

        # Contenedor con bordes para cada producto
        with st.container(border=True):
            st.subheader(prod)

            col1, col2 = st.columns([1, 2], gap="medium")
            with col1:
                st.metric("Precio Promedio", f"${precio_medio:,.2f}", f"{variaciones['Delta_Precio']:.2f}%")
                st.metric("Margen Promedio", f"{margen_medio:.2f}%", f"{variaciones['Delta_Margen']:.2f}%")
                st.metric("Total Unidades Vendidas", f"{total_unidades:,.0f}", f"{variaciones['Delta_Unidades']:.2f}%")
            with col2:
                grafico = generar_grafico(data, prod)
                st.pyplot(grafico)

def ejecutar_app():
    datos_cargados = cargar_archivo()
    if datos_cargados is not None:
        datos_cargados = calcular_metrica(datos_cargados)
        datos_cargados = calcular_variaciones(datos_cargados)  # Cálculo de las variaciones
        mostrar_datos(datos_cargados)
    else:
        st.title("Por favor sube un archivo CSV desde la barra lateral.")
        mostrar_info_usuario()  # Mostrar información del usuario si no hay datos

if __name__ == "__main__":
    ejecutar_app()
