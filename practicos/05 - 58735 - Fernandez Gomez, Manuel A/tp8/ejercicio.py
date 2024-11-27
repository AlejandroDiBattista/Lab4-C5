import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# url ='https://agulab-4.streamlit.app'


with st.sidebar:
    uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
    sucursal_seleccionada = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        sucursales = ["Todas"] + sorted(df['Sucursal'].unique().tolist())
        sucursal_seleccionada = st.selectbox("Seleccionar ", sucursales)

if sucursal_seleccionada:
    if sucursal_seleccionada == "Todas":
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de la {sucursal_seleccionada}")

if uploaded_file:
    df['Fecha'] = pd.to_datetime(df.rename(columns={'Año': 'year', 'Mes': 'month'})[['year', 'month']].assign(day=1))

    if sucursal_seleccionada != "Todas":
        df = df[df['Sucursal'] == sucursal_seleccionada]

    productos = df['Producto'].unique()
    for producto in productos:
        with st.container():

            df_producto = df[df['Producto'] == producto]
    
            df_producto['Precio_unitario'] = df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']
    
            precio_promedio_actual = df_producto['Precio_unitario'].mean()  # Promedio del precio unitario
            margen_promedio_actual = ((df_producto['Ingreso_total'] - df_producto['Costo_total']) / df_producto['Ingreso_total']).mean()
            unidades_vendidas_actual = df_producto['Unidades_vendidas'].sum()
    
            df_producto['Año'] = df_producto['Fecha'].dt.year
            df_anterior = df_producto[df_producto['Año'] == df_producto['Año'].max() - 1]
            if not df_anterior.empty:
                precio_promedio_anterior = df_anterior['Precio_unitario'].mean()
                margen_promedio_anterior = ((df_anterior['Ingreso_total'] - df_anterior['Costo_total']) / df_anterior['Ingreso_total']).mean()
                unidades_vendidas_anterior = df_anterior['Unidades_vendidas'].sum()
    
                delta_precio = ((precio_promedio_actual - precio_promedio_anterior) / precio_promedio_anterior) * 100
                delta_margen = ((margen_promedio_actual - margen_promedio_anterior) / margen_promedio_anterior) * 100
                delta_unidades = ((unidades_vendidas_actual - unidades_vendidas_anterior) / unidades_vendidas_anterior) * 100
            else:
                delta_precio, delta_margen, delta_unidades = None, None, None
    
            col1, col_sep, col2 = st.columns([1, 0.1, 3])
            with col1:
                st.subheader(f"Producto: {producto}")
                st.metric("Precio Promedio", f"${precio_promedio_actual:.2f}", delta=f"{delta_precio:.2f}%" if delta_precio is not None else "N/A")
                st.metric("Margen Promedio", f"{margen_promedio_actual * 100:.2f}%", delta=f"{delta_margen:.2f}%" if delta_margen is not None else "N/A")
                st.metric("Unidades Vendidas", f"{int(unidades_vendidas_actual):,}", delta=f"{delta_unidades:.2f}%" if delta_unidades is not None else "N/A")
    
            with col2:
                df_producto = df_producto.sort_values('Fecha')
                plt.figure(figsize=(10, 5))
                plt.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], label=producto)
    
                X = np.array(range(len(df_producto['Fecha']))).reshape(-1, 1)
                y = df_producto['Unidades_vendidas'].values
                model = LinearRegression().fit(X, y)
                tendencia = model.predict(X)
    
                plt.plot(df_producto['Fecha'], tendencia, color='red', label="Tendencia")
                plt.title("Evolución de Ventas Mensual")
                plt.xlabel("Fecha")
                plt.ylabel("Unidades Vendidas")
                plt.legend()
    
                st.pyplot(plt)
                plt.clf()
    
            st.markdown("---")


            st.markdown(
                """
                <style>
                .footer {
                    position: fixed;
                    bottom: 0;
                    right: 0;
                    color: gray;
                    padding: 10px;
                    font-size: 12px;
                }
                </style>
                <div class="footer">Manuel Agustin Fernandez Gomez - 58735</div>
                """,
                unsafe_allow_html=True
                )