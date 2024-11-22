import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
        "xtick.top": False,
        "ytick.left": False,
        "ytick.right": False,

        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59076')
        st.markdown('**Nombre:** Gomez Martinez Matias Leonel')
        st.markdown('**Comisión:** C5')


def main():

    mostrar_informacion_alumno()

    class Producto:
        def __init__(self, nombre, datos):
            self.nombre = nombre
            self.datos = datos

        def tarjeta(self):
            self.datos = self.datos.copy()

            df_fecha = pd.DataFrame({'year': self.datos['Año'], 'month': self.datos['Mes'], 'day': 1})
            self.datos['fecha'] = pd.to_datetime(df_fecha)

            precio_promedio = self.datos['Ingreso_total'].sum() / self.datos['Unidades_vendidas'].sum()
            margen_promedio = (self.datos['Ingreso_total'].sum() - self.datos['Costo_total'].sum()) / self.datos['Ingreso_total'].sum() * 100
            unidades_totales = self.datos['Unidades_vendidas'].sum()

            self.datos = self.datos.sort_values(by='fecha')

            resumen_mensual = self.datos.groupby('fecha').agg({
                'Unidades_vendidas': 'sum',
                'Ingreso_total': 'sum',
                'Costo_total': 'sum'
            }).reset_index()

            resumen_mensual['var_unidades'] = resumen_mensual['Unidades_vendidas'].pct_change() * 100
            resumen_mensual['var_ingreso'] = resumen_mensual['Ingreso_total'].pct_change() * 100
            resumen_mensual['var_costo'] = resumen_mensual['Costo_total'].pct_change() * 100

            with st.container(border=True):
                st.subheader(self.nombre)
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Precio Promedio", f"${precio_promedio:,.2f}", f"{resumen_mensual['var_ingreso'].iloc[-1]:.2f}%")
                    st.metric("Margen Promedio", f"{margen_promedio:.2f}%", f"{resumen_mensual['var_costo'].iloc[-1]:.2f}%")
                    st.metric("Unidades Vendidas", f"{unidades_totales:,.0f}", f"{resumen_mensual['var_unidades'].iloc[-1]:.2f}%")

                with col2:
                    set_seaborn_style('Arial', "#042940", "#005C53", "#D6D58E")

                    fig, ax = plt.subplots(figsize=(6, 4))

                    ventas_mensuales = self.datos.groupby('fecha')['Unidades_vendidas'].sum().reset_index()
                    X = np.arange(len(ventas_mensuales)).reshape(-1, 1)
                    y = ventas_mensuales['Unidades_vendidas'].values.reshape(-1, 1)

                    modelo = LinearRegression()
                    modelo.fit(X, y)
                    tendencia = modelo.predict(X)

                    sns.lineplot(
                        data=ventas_mensuales,
                        x='fecha', y='Unidades_vendidas',
                        label=f"{self.nombre} - Ventas", ax=ax
                    )
                    ax.plot(ventas_mensuales['fecha'], tendencia, color="red", linestyle="--", label="Tendencia")

                    ax.set_title(f"Evolución de Ventas Mensual - {self.nombre}", fontsize=10)
                    ax.set_xlabel("Fecha", fontsize=8)
                    ax.set_ylabel("Unidades Vendidas", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.legend(fontsize=7)
                    ax.grid(True)
                    ax.set_ylim(0, max(ventas_mensuales['Unidades_vendidas'].values) * 1.1)

                    st.pyplot(fig)

    l = ['Todos', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
    with st.sidebar:
        st.header("Cargar archivo de datos")
        datos = st.file_uploader("Subir archivo CSV", type=["csv"])

        if datos is not None:
            df = pd.read_csv(datos)
        else:
            st.write("No se seleccionó un archivo")
            st.stop()

        suc = st.selectbox('🏢 Seleccionar Sucursal', l)

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
