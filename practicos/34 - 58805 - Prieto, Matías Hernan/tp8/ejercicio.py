import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://rtje9hegpctq33mfrcq6ed.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.805')
        st.markdown('**Nombre:** Prieto Matías')
        st.markdown('**Comisión:** C5')

mostrar_informacion_alumno()


st.sidebar.title('Cargar archivo de datos')
csv = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

if csv is None:
    st.sidebar.warning("Subí un archivo CSV para comenzar.")
    st.stop()

data = pd.read_csv(csv)


required_columns = ["Sucursal", "Producto", "Año", "Mes", "Unidades_vendidas", "Ingreso_total", "Costo_total"]
if not set(required_columns).issubset(data.columns):
    st.error(f"El archivo debe contener las columnas: {', '.join(required_columns)}")
    st.stop()


data["Año-Mes"] = data["Año"].astype(str) + "-" + data["Mes"].astype(str).str.zfill(2)

sucursal = st.sidebar.selectbox(
    "Seleccionar Sucursal",
    options=["Todas"] + sorted(data["Sucursal"].unique())
)

if sucursal != "Todas":
    data = data[data["Sucursal"] == sucursal]
if sucursal == 'Todas':
    st.title(f"Datos de todas las Sucursales")
else:
    st.title(f"Datos de {sucursal}")
for producto in data["Producto"].unique():
    df_producto = data[data["Producto"] == producto]

    df_producto["Fecha"] = pd.to_datetime(
        df_producto["Año"].astype(str) + "-" + df_producto["Mes"].astype(str).str.zfill(2)
    )
    df_producto = df_producto.sort_values(by="Fecha")
    

    unidades_total = df_producto["Unidades_vendidas"].sum()
  
    df_producto['Precio_promedio'] = df_producto["Ingreso_total"] / df_producto["Unidades_vendidas"]
    precio_promedio = df_producto["Precio_promedio"].mean()
    df_producto["Margen_promedio"] = (df_producto["Ingreso_total"] - df_producto["Costo_total"]) / df_producto["Ingreso_total"]
    margen_promedio = df_producto["Margen_promedio"].mean()

    df_producto["Precio_Promedio"] = np.where(
    df_producto["Unidades_vendidas"] > 0,
    df_producto["Ingreso_total"] / df_producto["Unidades_vendidas"],
    0)

    df_producto["Margen_Promedio"] = np.where(
    df_producto["Ingreso_total"] > 0,
    (df_producto["Ingreso_total"] - df_producto["Costo_total"]) / df_producto["Ingreso_total"],
    0)

    df_producto["Cambio_Precio"] = df_producto["Precio_Promedio"].pct_change() * 100  
    df_producto["Cambio_Margen"] = df_producto["Margen_Promedio"].pct_change() * 100
    df_producto["Cambio_Unidades"] = df_producto["Unidades_vendidas"].pct_change() * 100

    with st.container(border=True):
        col1, col2 = st.columns([1, 3])  
        with col1:
            st.subheader(producto)
            st.metric("Precio Promedio", f"${precio_promedio:,.0f}".replace(',', '.'), f"{df_producto['Cambio_Precio'].iloc[-1]:.2f}%")
            st.metric("Margen Promedio", f"{margen_promedio * 100:.0f}%".replace(',', '.'), f"{df_producto['Cambio_Margen'].iloc[-1]:.2f}%")
            st.metric("Unidades Vendidas", f"{unidades_total:,}".replace(',', '.'), f"{df_producto['Cambio_Unidades'].iloc[-1]:.2f}%")
        with col2:
            fig, ax = plt.subplots(figsize=(18, 10)) 
            ax.plot(
                df_producto["Fecha"],  
                df_producto["Unidades_vendidas"],
                label="Unidades Vendidas",
                color="blue"
            )
            z = np.polyfit(range(len(df_producto)), df_producto["Unidades_vendidas"], 1)  
            tendencia = np.poly1d(z)(range(len(df_producto)))  
            ax.plot(
                df_producto["Fecha"],
                tendencia,
                label="Tendencia",
                linestyle="--",
                color="red",
            )
            ax.set_xticks(df_producto["Fecha"]) 
            ax.set_xticklabels(df_producto["Fecha"].dt.strftime('%Y-%m'), rotation=50, ha="right") 
            ax.set_title(f"Evolución de Ventas - {producto}")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Unidades Vendidas")
            ax.legend()
            ax.grid(True)
            plt.tight_layout(pad=3.0)
            plt.subplots_adjust(top=2.80, bottom=1.80)  
            st.pyplot(fig)