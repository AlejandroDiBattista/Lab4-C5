import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-58724.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.724')
        st.markdown('**Nombre:** Julian Lopez')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()

# Función para aplicar estilo con Seaborn
def set_seaborn_style(font_family, background_color, grid_color, text_color):
    sns.set_style({
        "axes.facecolor": background_color,
        "figure.facecolor": background_color,

        "grid.color": grid_color,
        "axes.edgecolor": grid_color,
        "axes.grid": True,
        "axes.axisbelow": True,
        
        "axes.labelcolor": text_color,
        "text.color": text_color,
        "font.family": font_family,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "xtick.bottom": False,
        "ytick.left": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })

# Clase Producto para manejar cada producto
class Producto:
    def __init__(self, nombre, datos):
        self.nombre = nombre
        self.datos = datos

    def tarjeta(self):
        # Cálculo de métricas
        precio_promedio = self.datos['Ingreso_total'].sum() / self.datos['Unidades_vendidas'].sum()
        margen_promedio = (self.datos['Ingreso_total'].sum() - self.datos['Costo_total'].sum()) / self.datos['Ingreso_total'].sum() * 100
        unidades_totales = self.datos['Unidades_vendidas'].sum()

        # Cálculo de variaciones reales
        precio_anterior = self.datos['Ingreso_total'].iloc[:-1].sum() / self.datos['Unidades_vendidas'].iloc[:-1].sum()
        delta_precio = ((precio_promedio - precio_anterior) / precio_anterior) * 100 if precio_anterior else 0

        margen_anterior = (self.datos['Ingreso_total'].iloc[:-1].sum() - self.datos['Costo_total'].iloc[:-1].sum()) / self.datos['Ingreso_total'].iloc[:-1].sum() * 100
        delta_margen = ((margen_promedio - margen_anterior) / margen_anterior) * 100 if margen_anterior else 0

        unidades_anterior = self.datos['Unidades_vendidas'].iloc[:-1].sum()
        delta_unidades = ((unidades_totales - unidades_anterior) / unidades_anterior) * 100 if unidades_anterior else 0

        # Conversión de fecha
        df_fecha = pd.DataFrame({'year': self.datos['Año'], 'month': self.datos['Mes'], 'day': 1})
        self.datos['fecha'] = pd.to_datetime(df_fecha)

        # Contenedor de presentación
        with st.container():
            st.markdown(f"### {self.nombre}")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Precio Promedio", f"${precio_promedio:,.0f}", delta=f"{delta_precio:.2f}%")
                st.metric("Margen Promedio", f"{margen_promedio:.0f}%", delta=f"{delta_margen:.2f}%")
                st.metric("Unidades Vendidas", f"{unidades_totales:,.0f}", delta=f"{delta_unidades:.2f}%")

            with col2:
                # Estilo del gráfico
                set_seaborn_style('Arial', "#f5f5f5", "#cccccc", "#333333")
                fig, ax = plt.subplots(figsize=(8, 4))

                # Datos de ventas mensuales y tendencia
                ventas_mensuales = self.datos.groupby('fecha')['Unidades_vendidas'].sum().reset_index()
                X = np.arange(len(ventas_mensuales)).reshape(-1, 1)
                y = ventas_mensuales['Unidades_vendidas'].values.reshape(-1, 1)
                modelo = LinearRegression()
                modelo.fit(X, y)
                tendencia = modelo.predict(X)

                sns.lineplot(data=ventas_mensuales, x='fecha', y='Unidades_vendidas', ax=ax, label="Unidades Vendidas", color="#4CAF50")
                ax.plot(ventas_mensuales['fecha'], tendencia, color="red", linestyle="--", label="Tendencia")

                ax.set_title(f"Evolución de Ventas - {self.nombre}", fontsize=12)
                ax.set_xlabel("Fecha", fontsize=10)
                ax.set_ylabel("Unidades Vendidas", fontsize=10)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=8)
                ax.grid(True)
                st.pyplot(fig)

# Función principal
def main():
    sucursales = ['Todos', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']

    with st.sidebar:
        st.header("Cargar archivo de datos")
        archivo = st.file_uploader("Subir archivo CSV", type=["csv"])
        sucursal = st.selectbox("Seleccionar Sucursal", sucursales)

        if archivo is not None:
            datos = pd.read_csv(archivo)
        else:
            st.warning("Por favor, sube un archivo CSV para continuar.")
            st.stop()

        if sucursal != 'Todos':
            datos = datos[datos['Sucursal'] == sucursal]

    if sucursal == 'Todos':
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de la {sucursal}")

    for producto in datos['Producto'].unique():
        datos_producto = datos[datos['Producto'] == producto]
        prod = Producto(producto, datos_producto)
        prod.tarjeta()

main()