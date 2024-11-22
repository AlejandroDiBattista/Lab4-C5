import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://lab4-c5-ktervsfjfrqdhsap6urpf6.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58.687')
        st.markdown('**Nombre:** Sofia Nahir Jadur')
        st.markdown('**Comisión:** C5')


def cargar_archivo():
    st.sidebar.title("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
    if archivo is not None:
        datos = pd.read_csv(archivo)
        return datos
    return None


def calcular_resumen(datos, sucursal):
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]

    productos = datos["Producto"].unique()
    resumen = []
    
    for producto in productos:
        data_prod = datos[datos["Producto"] == producto]
        precio_prom = data_prod["Ingreso_total"].sum() / data_prod["Unidades_vendidas"].sum()
        margen_prom = (data_prod["Ingreso_total"].sum() - data_prod["Costo_total"].sum()) / data_prod["Ingreso_total"].sum()
        unidades_total = data_prod["Unidades_vendidas"].sum()

    
        cambios = {
            "Precio": np.random.uniform(-10, 10),
            "Margen": np.random.uniform(-5, 5),
            "Unidades": np.random.uniform(-15, 15)
        }

        resumen.append({
            "Producto": producto,
            "Precio Promedio": precio_prom,
            "Margen Promedio": margen_prom,
            "Unidades Vendidas": unidades_total,
            "Cambio Precio": cambios["Precio"],
            "Cambio Margen": cambios["Margen"],
            "Cambio Unidades": cambios["Unidades"]
        })
    
    return pd.DataFrame(resumen)


def mostrar_datos(columna, etiqueta, valor, cambio):
    delta_color = "normal" if cambio > 0 else "inverse"
    formato_valor = f"${valor:,.2f}" if "Precio" in etiqueta else f"{valor:.0%}" if "Margen" in etiqueta else f"{valor:,.0f}"
    columna.metric(etiqueta, formato_valor, f"{cambio:+.2f}%")


def generar_grafico(data, producto):
    data_producto = data[data["Producto"] == producto].copy()

    
    data_producto["Año"] = data_producto["Año"].fillna(0).astype(int)
    data_producto["Mes"] = data_producto["Mes"].fillna(0).astype(int)
    data_producto["Fecha"] = pd.to_datetime({"year": data_producto["Año"], "month": data_producto["Mes"], "day": 1})
    data_producto.sort_values("Fecha", inplace=True)
    
    
    indices = range(len(data_producto))
    data_producto["Tendencia"] = np.poly1d(np.polyfit(indices, data_producto["Unidades_vendidas"], 1))(indices)

    
    plt.figure(figsize=(12, 6))
    plt.plot(data_producto["Fecha"], data_producto["Unidades_vendidas"], label="Unidades Vendidas")
    plt.plot(data_producto["Fecha"], data_producto["Tendencia"], label="Tendencia", linestyle="--")
    plt.title(f"Evolución de Ventas - {producto}")
    plt.xlabel("Año-Mes")
    plt.ylabel("Unidades Vendidas")
    plt.grid(linestyle="--", linewidth=0.5)
    plt.legend()
    st.pyplot(plt)


def ejecutar_app():
    datos = cargar_archivo()

    if datos is None:
        st.title("Por favor, sube un archivo CSV para analizar.")
        mostrar_informacion_alumno()
    else:
        sucursal_seleccionada = st.sidebar.selectbox(
            "Seleccionar Sucursal",
            ["Todas"] + list(datos["Sucursal"].unique())
        )
        
        if sucursal_seleccionada == "Todas":
            st.title("Datos de Todas las Sucursales")
        else:
            st.title(f"Datos de la {sucursal_seleccionada}")

        resumen = calcular_resumen(datos, sucursal_seleccionada)

        for _, fila in resumen.iterrows():
            with st.container(border=True):
                col_izq, col_der = st.columns([1, 2])

                with col_izq:
                    st.subheader(fila["Producto"])
                    mostrar_datos(col_izq, "Precio Promedio", fila["Precio Promedio"], fila["Cambio Precio"])
                    mostrar_datos(col_izq, "Margen Promedio", fila["Margen Promedio"], fila["Cambio Margen"])
                    mostrar_datos(col_izq, "Unidades Vendidas", fila["Unidades Vendidas"], fila["Cambio Unidades"])

                with col_der:
                    generar_grafico(datos, fila["Producto"])

if __name__ == "__main__":
    ejecutar_app()
