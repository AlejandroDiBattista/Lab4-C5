import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import itertools

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58895-flores.streamlit.app/'

def mostrar_datos_estudiante():
    st.markdown("""
    ### Sube un archivo CSV.
    """)
    with st.container(border=True):
        st.markdown('**Legajo:** 58895')
        st.markdown('**Nombre:** Flores Gonzalez Tomas Agustin')
        st.markdown('**Comisión:** C5')

class AnalisisVentas:
    def __init__(self, datos_ventas):
        self.datos = datos_ventas
        self.datos_por_producto = self.procesar_datos()

    def procesar_datos(self):
        return {
            producto: self.datos[self.datos['Producto'] == producto]
            for producto in self.datos['Producto'].unique()
        }

    def promedio_precio(self, datos_producto):
        return reduce(
            lambda acumulador, valor: acumulador + valor,
            datos_producto['Ingreso_total'] / datos_producto['Unidades_vendidas']
        ) / len(datos_producto)

    def variacion_precio(self, datos_producto):
        """Calcula la variación porcentual del precio promedio."""
        precios_anuales = datos_producto.groupby('Año').apply(
            lambda x: np.mean(x['Ingreso_total'] / x['Unidades_vendidas'])
        )
        return np.mean(np.diff(precios_anuales) / precios_anuales[:-1]) * 100

    def margen_promedio(self, datos_producto):
        """Calcula el margen promedio de ganancia."""
        ingresos = datos_producto['Ingreso_total']
        costos = datos_producto['Costo_total']
        margenes = (ingresos - costos) / ingresos * 100
        return np.mean(margenes)

    def variacion_margen(self, datos_producto):
        """Calcula la variación porcentual del margen promedio."""
        margenes_anuales = datos_producto.groupby('Año').apply(
            lambda x: np.mean((x['Ingreso_total'] - x['Costo_total']) / x['Ingreso_total'] * 100)
        )
        return np.mean(np.diff(margenes_anuales) / margenes_anuales[:-1]) * 100

    def total_unidades(self, datos_producto):
        """Calcula el total de unidades vendidas."""
        return np.sum(datos_producto['Unidades_vendidas'])

    def variacion_unidades(self, datos_producto):
        """Calcula la variación porcentual del total de unidades vendidas."""
        unidades_anuales = datos_producto.groupby('Año')['Unidades_vendidas'].sum()
        pares_unidades = list(itertools.pairwise(unidades_anuales))
        variaciones = [((b - a) / a) * 100 for a, b in pares_unidades]
        return np.mean(variaciones)

def generar_grafico(datos_producto, producto):
    """Genera un gráfico de tendencia de ventas."""
    ventas_mensuales = datos_producto.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    x_vals = np.arange(len(ventas_mensuales))
    y_vals = ventas_mensuales['Unidades_vendidas']

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200, facecolor='#f9f9f9')
    ax.plot(x_vals, y_vals, label=producto, color='blue', marker='o', linewidth=3)

    # Línea de tendencia
    coef = np.polyfit(x_vals, y_vals, 1)
    tendencia = np.poly1d(coef)
    ax.plot(x_vals, tendencia(x_vals), linestyle='--', color='red', label='Tendencia', linewidth=2)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{row.Año}" if row.Mes == 1 else "" for row in ventas_mensuales.itertuples()])
    ax.set_title("Tendencia de Ventas", fontsize=14, color='#333')
    ax.set_xlabel("Tiempo (Año-Mes)", fontsize=12, color='#666')
    ax.set_ylabel("Unidades Vendidas", fontsize=12, color='#666')
    ax.grid(alpha=0.4, linestyle='--')

    ax.legend(loc='best', frameon=True)
    return fig

def main():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

    if archivo:
        datos = pd.read_csv(archivo)
        analisis = AnalisisVentas(datos)

        sucursales = ["Todas"] + list(datos['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal_seleccionada != "Todas":
            datos = datos[datos['Sucursal'] == sucursal_seleccionada]
            st.title(f"Datos de Todas las Sucursales" if sucursal_seleccionada == "Todas" else f"Datos de {sucursal_seleccionada}")
        else:
            st.title(f"Datos de {sucursal_seleccionada} las Sucursales")

        for producto in datos['Producto'].unique():
            st.subheader(f"{producto}")
            datos_producto = datos[datos['Producto'] == producto]

            promedio_precio = analisis.promedio_precio(datos_producto)
            variacion_precio = analisis.variacion_precio(datos_producto)
            margen_prom = analisis.margen_promedio(datos_producto)
            variacion_margen = analisis.variacion_margen(datos_producto)
            total_unidades = analisis.total_unidades(datos_producto)
            variacion_unidades = analisis.variacion_unidades(datos_producto)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Precio Promedio", f"${promedio_precio:,.2f}", f"{variacion_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_prom:.2f}%", f"{variacion_margen:.2f}%")
                st.metric("Unidades Totales", f"{total_unidades:,}", f"{variacion_unidades:.2f}%")

            with col2:
                fig = generar_grafico(datos_producto, producto)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
    mostrar_datos_estudiante()
