import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = https://tp8-59071.streamlit.app/

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59071')
        st.markdown('**Nombre:** Micaela Medina')
        st.markdown('**Comisión:** C5')

def calcular_metricas(df):
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


def graficar_evolucion(df, producto):
    datos_producto = df[df['Producto'] == producto].groupby(['Año', 'Mes']).sum().reset_index()

    if 'Año' not in datos_producto.columns or 'Mes' not in datos_producto.columns:
        st.error("Las columnas 'Año' o 'Mes' faltan en los datos.")
        return

    datos_producto = datos_producto.dropna(subset=['Año', 'Mes'])
    try:
        datos_producto['Año'] = datos_producto['Año'].astype(int)
        datos_producto['Mes'] = datos_producto['Mes'].astype(int)
    except ValueError:
        st.error("Error al convertir 'Año' y 'Mes' a enteros.")
        return
    try:
        datos_producto['Fecha'] = pd.to_datetime(
            {
                'year': datos_producto['Año'],
                'month': datos_producto['Mes'],
                'day': 1
            },
            errors='coerce'
        )
        datos_producto = datos_producto.dropna(subset=['Fecha'])
    except Exception as e:
        st.error(f"Error al crear la columna de fecha: {e}")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        datos_producto['Fecha'], 
        datos_producto['Unidades_vendidas'], 
        label=producto, 
        color="blue", 
        linewidth=2  
    )
    
    z = np.polyfit(range(len(datos_producto)), datos_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    plt.plot(
        datos_producto['Fecha'], 
        p(range(len(datos_producto))), 
        "--", 
        label="Tendencia", 
        color="red", 
        linewidth=2  
    )

    plt.grid(
        which='major',  
        color='gray',   
        linestyle='-',
        linewidth=0.8  
    )
    
    plt.minorticks_on() 
    plt.grid(
        which='minor', 
        color='lightgray',  
        linestyle=':',      
        linewidth=0.5      
    )

 
    plt.title(f"Evolución de Ventas Mensual - {producto}")
    plt.xlabel("Fecha")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    plt.tight_layout()  
    st.pyplot(plt)

def main():
    st.title("Datos de Todas Las Sucursales")
    
    st.sidebar.header("Cargar archivo de datos")
    archivo_csv = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

    if archivo_csv is None:
        mostrar_informacion_alumno()

    if archivo_csv is not None:
        df = pd.read_csv(archivo_csv)

        columnas_requeridas = {'Año', 'Mes', 'Producto', 'Ingreso_total', 'Unidades_vendidas', 'Costo_total'}
        if not columnas_requeridas.issubset(df.columns):
            st.error(f"El archivo cargado no contiene las columnas necesarias: {columnas_requeridas}")
            return

        df = calcular_metricas(df)

        sucursales = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"] + df['Sucursal'].unique().tolist())
        if sucursales != "Todas":
            df = df[df['Sucursal'] == sucursales]

        # Recalcular variaciones después de aplicar el filtro por sucursal
        variaciones = calcular_variaciones(df)

        productos = df['Producto'].unique()
        for producto in productos:
            
            with st.container():
                st.markdown("---") 
                st.subheader(producto)
                
                datos_producto = df[df['Producto'] == producto]
                precio_promedio = datos_producto['Precio Promedio'].mean()
                margen_promedio = datos_producto['Margen'].mean()
                unidades_vendidas = datos_producto['Unidades_vendidas'].sum()

                variacion_precio = variaciones[producto]["variacion_precio_promedio"]
                variacion_margen = variaciones[producto]["variacion_margen_promedio"]
                variacion_unidades = variaciones[producto]["variacion_unidades_vendidas"]

                precio_promedio_str = f"${precio_promedio:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                margen_promedio_str = f"{margen_promedio:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
                unidades_vendidas_str = f"{int(unidades_vendidas):,}".replace(",", ".")

                col1, col2 = st.columns([1, 2])  

                with col1:
                    st.metric("Precio Promedio", precio_promedio_str, delta=f"{variacion_precio:+.2f}%")
                    st.metric("Margen Promedio", margen_promedio_str, delta=f"{variacion_margen:+.2f}%")
                    st.metric("Unidades Vendidas", unidades_vendidas_str, delta=f"{variacion_unidades:+.2f}%")

                with col2:
                    graficar_evolucion(df, producto)


if __name__ == "__main__":
    main()