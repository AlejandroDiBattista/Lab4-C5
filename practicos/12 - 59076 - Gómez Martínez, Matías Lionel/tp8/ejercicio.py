import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59076.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59076')
        st.markdown('**Nombre:** Gomez Martinez Matias Leonel')
        st.markdown('**Comisión:** C5')


def main():

    

    class Producto:
        def __init__(self, nombre, datos):
            self.nombre = nombre
            self.datos = datos

        def tarjeta(self):
            self.datos = self.datos.copy()

            datos_producto['Ppromedio'] = datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
            promedioP = datos_producto['Ppromedio'].mean()
            promedioPaño = datos_producto.groupby('Año')['Ppromedio'].mean()
            variacion_promedio = promedioPaño.pct_change().mean() * 100
            
            
            datos_producto['Margen'] = ((datos_producto['Ingreso_total'] - datos_producto['Costo_total']) / datos_producto['Ingreso_total']) * 100
            promedioM = datos_producto['Margen'].mean()
            promedioMaño = datos_producto.groupby('Año')['Margen'].mean()
            variacion_margen = promedioMaño.pct_change().mean() * 100
            
            unidades = datos_producto['Unidades_vendidas'].sum()
            unidadesAño = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
            variacion_unidades = unidadesAño.pct_change().mean() * 100

            with st.container(border=True):
                st.subheader(self.nombre)
                col1, col2 = st.columns([0.25, 0.75])

                with col1:
                    st.metric("Precio Promedio", f"${promedioP:,.2f}", f"{variacion_promedio:.2f}%")
                    st.metric("Margen Promedio", f"{promedioM:.2f}%", f"{variacion_margen:.2f}%")
                    st.metric("Unidades Vendidas", f"{unidades:,.0f}", f"{variacion_unidades:.2f}%")

                with col2:

                    ventas = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(len(ventas)), ventas['Unidades_vendidas'], label=f'{producto} - Ventas')
                    
                    x = np.arange(len(ventas))
                    y = ventas['Unidades_vendidas']
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)

                    ax.plot(x, p(x), linestyle='--', color='red', label='Tendencia')
                    
                    ax.set_title('Evolución de Ventas Mensual - '+producto)
                    ax.set_xlabel('Año-Mes')
                    ax.set_xticks(range(len(ventas)))
                    
                    etiquetas = []
                    for i, row in enumerate(ventas.itertuples()):
                        if row.Mes == 1:
                            etiquetas.append(f"{row.Año}")
                        else:
                            etiquetas.append("")
                    ax.set_xticklabels(etiquetas)
                    ax.set_ylabel('Unidades Vendidas')
                    ax.set_ylim(0, None)
                    ax.legend()
                    ax.grid(True)

                    st.pyplot(fig)

    l = ['Todos', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']

    st.sidebar.header("Cargar archivo de datos")
    datos = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

    if datos is not None:
        df = pd.read_csv(datos)
    else:
        mostrar_informacion_alumno()
        st.write("No se seleccionó un archivo")
        st.stop()

    suc = st.sidebar.selectbox('🏢 Seleccionar Sucursal', l)

    if suc == 'Todos':
        df_filtrado = df
    else:
        df_filtrado = df[df['Sucursal'] == suc]

    if suc == 'Todos':
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de la {suc}")

    for producto in df_filtrado['Producto'].unique():
        datos_producto = df_filtrado[df_filtrado['Producto'] == producto]
        prod = Producto(producto, datos_producto)
        prod.tarjeta()


if __name__ == "__main__":
    st.set_page_config(
        page_title="TP8-59076",
        page_icon=":chart_with_upwards_trend:",
        layout='wide'
    )
    main()
