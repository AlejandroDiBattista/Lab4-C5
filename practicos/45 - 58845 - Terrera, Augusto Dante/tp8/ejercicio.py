import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from matplotlib.ticker import MultipleLocator


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
url = 'https://tp8-58845-terreraaugusto.streamlit.app/'


def mostrar_informacion_alumno():
    st.header("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container(border=True):
        st.markdown('**Legajo:** 58.845')
        st.markdown('**Nombre:** Terrera Augusto Dante')
        st.markdown('**Comisión:** C5')
        




st.sidebar.header("Cargar archivo de datos")
archivo_subido = st.sidebar.file_uploader("Subir archivo CSV", type= "csv")

if archivo_subido:
    try:
        box_seleccionado = st.sidebar.selectbox("Seleccione Sucursal", ["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"])

        df = pd.read_csv(archivo_subido)

        if box_seleccionado != "Todas":
            df = df[df['Sucursal'] == box_seleccionado]

        if box_seleccionado == "Todas":
            st.title("Datos de Todas las Sucursales")
        else:
            st.title(f"Datos de {box_seleccionado}")

        df['Precio Promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
        df['Margen Promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
        df['Unidades Vendidas'] = df['Unidades_vendidas']

        df['Cambio Precio'] = df.groupby('Producto')['Precio Promedio'].pct_change() * 100
        df['Cambio Unidades'] = df.groupby('Producto')['Unidades Vendidas'].pct_change() * 100

        df['Cambio Precio'].fillna(0, inplace=True)
        df['Cambio Unidades'].fillna(0, inplace=True)

        datos = df.groupby('Producto').agg(
            {
                'Precio Promedio' : 'mean',
                'Margen Promedio' : 'mean',
                'Unidades Vendidas' : 'sum',
                'Cambio Precio' : 'last',
                'Cambio Unidades' : 'last',
            }).reset_index()

        datos['Precio Promedio'] = datos['Precio Promedio'].apply(lambda x: np.floor(x * 100) / 100)
        datos['Margen Promedio'] = datos['Margen Promedio'].apply(lambda x: np.floor(x * 100) / 100)

        for _, row in datos.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1,2])
                with col1:
                    st.subheader(f"{row['Producto']}")
                    st.metric("Precio Promedio", f"${row['Precio Promedio']:.2f}", f"{row['Cambio Precio']:.2f}%")
                    st.metric("Margen Promedio", f"{row['Margen Promedio']*100:.0f}%", f"{row['Cambio Unidades']:.2f}%")
                    st.metric("Unidades Vendidas", f"{row['Unidades Vendidas']:,}", f"{row['Cambio Unidades']:.2f}%")
                
                with col2:
                    
                    df_producto = df[df['Producto'] == row['Producto']]

                    df_producto['Año-Mes'] = df_producto['Año'].astype(str) + '-' + df_producto['Mes'].astype(str).str.zfill(2)
                    df_producto['Año-Mes'] = pd.to_datetime(df_producto['Año-Mes'], format='%Y-%m')

                    df_producto["Fecha_Num"] = (df_producto["Año-Mes"] - df_producto["Año-Mes"].min()).dt.days
                    x = df_producto["Fecha_Num"].values
                    y = df_producto["Unidades Vendidas"].values

                    m = (np.sum(x * y) - len(x) * np.mean(x) * np.mean(y)) / (np.sum(x**2) - len(x) * np.mean(x)**2)
                    b = np.mean(y) - m * np.mean(x)
                    df_producto["Tendencia"] = m * x + b

                    df_producto['Unidades Vendidas Suavizadas'] = df_producto['Unidades Vendidas'].rolling(window=3, center=True).mean()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(df_producto["Año-Mes"], df_producto["Unidades Vendidas Suavizadas"], label=row["Producto"], color="blue", linewidth=1.5)
                    ax.plot(df_producto["Año-Mes"], df_producto["Tendencia"], label="Tendencia", color="red", linestyle="--", linewidth=1.5)

                    ax.grid(visible=True, which='both', color='gray', linestyle='-', linewidth=1, alpha=0.7)

                    ax.xaxis.set_minor_locator(MonthLocator(interval=1))
                    ax.yaxis.set_minor_locator(MultipleLocator(10000))  
                    
                    ax.set_title("Evolución de Ventas Mensual", fontsize=12, loc='center', pad=20)
                    ax.set_xlabel("Año-Mes", fontsize=10)
                    ax.set_ylabel("Unidades Vendidas", fontsize=10)

                    ax.legend()
                    st.markdown("<br><br>", unsafe_allow_html=True) 
                    st.pyplot(fig)
                    

    except Exception as e:
        st.error(f"Error al leer el archivo : {e}")
        
    
    
else:
    mostrar_informacion_alumno()

