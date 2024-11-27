import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://tp8-58686-gustavorodriguez.streamlit.app/'


def mostrar_info():
    st.write("-------------------")
    st.write("Legajo: 58.686")
    st.write("Nombre: Gustavo Rodriguez")
    st.write("Comisión: C5")
    st.write("-------------------")


def crear_grafico(tabla, producto):
    ventas_agrupadas = tabla.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ventas_agrupadas)), ventas_agrupadas['Unidades_vendidas'], label=producto, linewidth=2)
    
    eje_x = range(len(ventas_agrupadas))
    eje_y = ventas_agrupadas['Unidades_vendidas']
    coef_tendencia = np.polyfit(eje_x, eje_y, 1)
    linea_tendencia = np.poly1d(coef_tendencia)
    plt.plot(eje_x, linea_tendencia(eje_x), '--r', label='Tendencia')
    
    plt.title('Evolución de Ventas Mensual', fontsize=14)
    plt.xlabel('Meses', fontsize=12)
    plt.ylabel('Unidades Vendidas', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    etiquetas_meses = []
    for fila in ventas_agrupadas.itertuples():
        if fila.Mes == 1:
            etiquetas_meses.append(str(fila.Año))
        else:
            etiquetas_meses.append("")
    
    plt.xticks(range(len(ventas_agrupadas)), etiquetas_meses)
    plt.legend(fontsize=12)
    
    return plt.gcf()


def principal():
    st.set_page_config(layout="wide")
    st.sidebar.header("Subir Archivo")
    archivo_subido = st.sidebar.file_uploader("Pon tu archivo CSV aquí", type=["csv"])
    
    if archivo_subido is not None:
        tabla_datos = pd.read_csv(archivo_subido)
        lista_sedes = ["Todas"] + tabla_datos['Sucursal'].unique().tolist()
        sede_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", lista_sedes)
    
        if sede_seleccionada != "Todas":
            tabla_datos = tabla_datos[tabla_datos['Sucursal'] == sede_seleccionada]
            st.title(f"Datos de {sede_seleccionada}")
        else:
            st.title("Datos de Todas las Sucursales")
        lista_articulos = tabla_datos['Producto'].unique()
        for articulo in lista_articulos:
            with st.container(border=True):
                st.subheader(f"{articulo}")
                datos_articulo = tabla_datos[tabla_datos['Producto'] == articulo]
                datos_articulo['Precio_promedio'] = datos_articulo['Ingreso_total'] / datos_articulo['Unidades_vendidas']
                promedio_precio = datos_articulo['Precio_promedio'].mean()
                promedio_precio_anual = datos_articulo.groupby('Año')['Precio_promedio'].mean()
                variacion_precio_anual = promedio_precio_anual.pct_change().mean() * 100
                
                datos_articulo['Ganancia'] = datos_articulo['Ingreso_total'] - datos_articulo['Costo_total']
                datos_articulo['Margen'] = (datos_articulo['Ganancia'] / datos_articulo['Ingreso_total']) * 100
                promedio_margen = datos_articulo['Margen'].mean()
                promedio_margen_anual = datos_articulo.groupby('Año')['Margen'].mean()
                variacion_margen_anual = promedio_margen_anual.pct_change().mean() * 100
                
                promedio_unidades = datos_articulo['Unidades_vendidas'].mean()
                total_unidades_vendidas = datos_articulo['Unidades_vendidas'].sum()
                unidades_agrupadas_anual = datos_articulo.groupby('Año')['Unidades_vendidas'].sum()
                variacion_unidades_anual = unidades_agrupadas_anual.pct_change().mean() * 100
                
                columna_metrica, columna_visual = st.columns([0.25, 0.75])
                
                with columna_metrica:
                    st.metric(label="Precio Promedio", value=f"${promedio_precio:,.0f}".replace(",", "."), delta=f"{variacion_precio_anual:.2f}%")
                    st.metric(label="Margen Promedio", value=f"{promedio_margen:.0f}%".replace(",", "."), delta=f"{variacion_margen_anual:.2f}%")
                    st.metric(label="Unidades Vendidas", value=f"{total_unidades_vendidas:,.0f}".replace(",", "."), delta=f"{variacion_unidades_anual:.2f}%")
                
                with columna_visual:
                    grafico = crear_grafico(datos_articulo, articulo)
                    st.pyplot(grafico)
    else:
        st.write("### INFORMACIÓN")
        mostrar_info()

if __name__ == "__main__":
    principal()
