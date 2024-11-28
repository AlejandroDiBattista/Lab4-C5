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
    else:
        return None

def calcular_resumen(datos, sucursal):
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]

    datos = datos.dropna(subset=["Unidades_vendidas", "Ingreso_total", "Costo_total", "Año", "Mes"])
    datos = datos[(datos["Unidades_vendidas"] > 0) & (datos["Ingreso_total"] > 0)]
    datos["Año"] = datos["Año"].astype(int)
    datos["Mes"] = datos["Mes"].astype(int)

    datos["Precio_unitario"] = datos["Ingreso_total"] / datos["Unidades_vendidas"]
    datos["Margen_unitario"] = (datos["Ingreso_total"] - datos["Costo_total"]) / datos["Ingreso_total"]

    productos = datos["Producto"].unique()
    resumen = []

    for producto in productos:
        data_prod = datos[datos["Producto"] == producto]

        precio_promedio = data_prod["Precio_unitario"].mean()
        margen_promedio = data_prod["Margen_unitario"].mean()
        unidades_totales = data_prod["Unidades_vendidas"].sum()

        precio_anual = data_prod.groupby('Año')["Precio_unitario"].mean()
        margen_anual = data_prod.groupby('Año')["Margen_unitario"].mean()
        unidades_anuales = data_prod.groupby('Año')["Unidades_vendidas"].sum()

        cambio_precio = precio_anual.pct_change().mean() * 100
        cambio_margen = margen_anual.pct_change().mean() * 100
        cambio_unidades = unidades_anuales.pct_change().mean() * 100

        cambio_precio = cambio_precio if not np.isnan(cambio_precio) else 0
        cambio_margen = cambio_margen if not np.isnan(cambio_margen) else 0
        cambio_unidades = cambio_unidades if not np.isnan(cambio_unidades) else 0

        resumen.append({
            "Producto": producto,
            "Precio Promedio": precio_promedio,
            "Margen Promedio": margen_promedio,
            "Unidades Vendidas": unidades_totales,
            "Cambio Precio": cambio_precio,
            "Cambio Margen": cambio_margen,
            "Cambio Unidades": cambio_unidades
        })

    return pd.DataFrame(resumen)

def mostrar_datos(columna, etiqueta, valor, cambio):
    if "Precio" in etiqueta:
        formato_valor = f"${valor:,.0f}"
    elif "Margen" in etiqueta:
        formato_valor = f"{valor*100:.0f}%"
    elif "Unidades" in etiqueta:
        formato_valor = f"{valor:,.0f}"
    else:
        formato_valor = f"{valor}"
    
    columna.metric(etiqueta, formato_valor, f"{cambio:+.2f}%")

def generar_grafico(datos, producto):
    data_producto = datos[datos["Producto"] == producto].copy()

    data_producto["Año"] = pd.to_numeric(data_producto["Año"], errors="coerce").astype(int)
    data_producto["Mes"] = pd.to_numeric(data_producto["Mes"], errors="coerce").astype(int)
    data_producto = data_producto.dropna(subset=["Año", "Mes"])

    data_producto["Fecha"] = pd.to_datetime({
        "year": data_producto["Año"],
        "month": data_producto["Mes"],
        "day": 1
    }, errors="coerce")

    data_producto = data_producto.dropna(subset=["Fecha"])
    data_producto.sort_values("Fecha", inplace=True)

    ventas_mensuales = data_producto.groupby("Fecha")["Unidades_vendidas"].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(ventas_mensuales["Fecha"], ventas_mensuales["Unidades_vendidas"], label="Unidades Vendidas", marker="o")
    z = np.polyfit(range(len(ventas_mensuales)), ventas_mensuales["Unidades_vendidas"], 1)
    p = np.poly1d(z)
    plt.plot(ventas_mensuales["Fecha"], p(range(len(ventas_mensuales))), linestyle="--", color="red", label="Tendencia")
    plt.title(f"Evolución de Ventas - {producto}")
    plt.xlabel("Fecha")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def ejecutar_app():
    datos = cargar_archivo()

    if datos is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
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
