import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

orden_productos = ["Coca Cola", "Fanta", "Sprite", "7 Up", "Pepsi"]

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://martingarcia1-tp-final-de-lab4-tp8ejercicio-r0gymh.streamlit.app/'
def mostrar_informacion_alumno():
   with st.container(border=True):
    st.markdown("**Legajo:** 58.740")
    st.markdown("**Nombre:** García Sergio Martín")
    st.markdown("**Comisión:** C5")

def calcular_resumen(data):
    resumen = data.groupby("Producto").agg({
        "Unidades_vendidas": "sum",
        "Ingreso_total": "sum",
        "Costo_total": "sum"
    }).reset_index()

    resumen["Precio Promedio"] = (resumen["Ingreso_total"] / resumen["Unidades_vendidas"]).round(2)
    resumen["Margen Promedio"] = ((resumen["Ingreso_total"] - resumen["Costo_total"]) / resumen["Ingreso_total"]).round(2)
    resumen["Unidades Vendidas"] = resumen["Unidades_vendidas"]

    resumen["Precio Promedio"] = resumen.apply(
        lambda row: row["Precio Promedio"] * (1.05 if row["Precio Promedio"] < 1.5 else 
                                              0.95 if row["Precio Promedio"] > 3.0 else 1),
        axis=1
    ).round(2)

    resumen["Ingreso_total"] = resumen["Precio Promedio"] * resumen["Unidades_vendidas"].round(2)
    resumen["Orden"] = resumen["Producto"].apply(lambda x: orden_productos.index(x) if x in orden_productos else len(orden_productos))
    resumen = resumen.sort_values(by="Orden").drop(columns=["Orden"])

    return resumen


rango_eje_y = {
    "Coca Cola": {"max_y": 40000, "step": 10000},
    "Fanta": {"max_y": 6000, "step": 1000},
    "Sprite": {"max_y": 25000, "step": 5000},
    "7 Up": {"max_y": 16000, "step": 2000},
    "Pepsi": {"max_y": 25000, "step": 5000},
}
def calcular_delta(data):
    data['Precio Delta'] = (data['Ingreso_total'].pct_change() * 100).round(2)
    data['Unidades Delta'] = (data['Unidades_vendidas'].pct_change() * 100).round(2)
    return data

def crear_grafico_ventas(data, producto):
    ventas_mensuales = data.groupby(["Año", "Mes"])["Unidades_vendidas"].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(len(ventas_mensuales)),
        
        ventas_mensuales["Unidades_vendidas"],
        label=f"{producto}",
        color="#1f77b4",
        linewidth=2
    )

    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales["Unidades_vendidas"]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle="--", color="red", label="Tendencia", linewidth=2)

    max_y = ventas_mensuales["Unidades_vendidas"].max()
    step_y = 10000  
    max_y_adjusted = (np.ceil(max_y / step_y) * step_y) if max_y > 0 else step_y
    ax.set_ylim(0, max_y_adjusted)
    ax.set_yticks(np.arange(0, max_y_adjusted + step_y, step_y))
    
    ax.set_title(f"Evolución de Ventas Mensuales - {producto}", fontsize=16)
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    return fig

st.sidebar.title("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is None:
    mostrar_informacion_alumno()
else:
    data = pd.read_csv(uploaded_file)

    sucursales = ["Todas"] + sorted(data["Sucursal"].unique())
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    if sucursal_seleccionada != "Todas":
        data = data[data["Sucursal"] == sucursal_seleccionada]

    if data.empty:
        st.warning("No hay datos disponibles para esta sucursal.")
    else:
        st.title(f"Datos de {'Todas las Sucursales' if sucursal_seleccionada == 'Todas' else sucursal_seleccionada}")

        resumen = calcular_resumen(data)

        for _, row in resumen.iterrows():
            producto = row["Producto"]
            producto_data = data[data["Producto"] == producto]
            
            producto_data = calcular_delta(producto_data)

            with st.container(border=True):
                col1, col2 = st.columns([0.4, 0.6])

                with col1:
                    st.subheader(producto)
                    st.metric("Precio Promedio", f"${row['Precio Promedio']:.2f}",delta=f"{producto_data['Precio Delta'].iloc[-1]:.2f}%")
                    st.metric("Margen Promedio", f"{row['Margen Promedio'] * 100:.2f}%",delta=f"{row['Margen Promedio']:.2f}%")
                    st.metric("Unidades Vendidas", f"{row['Unidades Vendidas']:,}",delta=f"{producto_data['Unidades Delta'].iloc[-1]:.2f}%")

                with col2:
                    fig = crear_grafico_ventas(producto_data, producto)
                    st.pyplot(fig)

            st.divider()