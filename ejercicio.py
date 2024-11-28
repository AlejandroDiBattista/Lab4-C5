import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'URL = https://58951-ezequiel-robles.streamlit.app/'

def mostrar_informacion_alumno():
 st.title("Por favor, sube un archivo CSV desde la barra lateral.")
 with st.container(border=True):
      st.markdown('**Legajo:** 58.951')
      st.markdown('**Nombre:** Ezequiel Robles')
      st.markdown('**Comisión:** C5')

def procesar_archivo(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)

        df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
        df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')
        df = df.dropna(subset=['Año', 'Mes']).loc[df['Año'].between(2000, 2024)]
        df['Mes'] = df['Mes'].apply(lambda x: x if 1 <= x <= 12 else None)
        df = df.dropna(subset=['Mes'])
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2))

        sucursales_disponibles = ['Todas'] + df['Sucursal'].unique().tolist()
        sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales_disponibles)

        df_filtrado = df if sucursal_seleccionada == 'Todas' else df[df['Sucursal'] == sucursal_seleccionada]

        df_filtrado['Precio_promedio'] = df_filtrado['Ingreso_total'] / df_filtrado['Unidades_vendidas']
        df_filtrado['Margen_promedio'] = (df_filtrado['Ingreso_total'] - df_filtrado['Costo_total']) / df_filtrado['Ingreso_total']

       
        grouped = (
            df_filtrado.groupby('Producto')
            .agg(
                Unidades_vendidas=('Unidades_vendidas', 'sum'),
                Precio_promedio=('Precio_promedio', 'mean'),
                Margen_promedio=('Margen_promedio', 'mean')
            )
            .reset_index()
        )

        st.header(f"Datos de Todas las Sucursales {'- ' + sucursal_seleccionada if sucursal_seleccionada != 'Todas' else ''}")
        for _, row in grouped.iterrows():
         with st.container(border=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.subheader(row['Producto'])
                st.metric("Precio Promedio", f"${row['Precio_promedio']:.0f}", delta=f"${row['Precio_promedio'] * 0.1:.2f}")
                st.metric("Margen Promedio", f"{row['Margen_promedio'] * 100:.2f}%", delta=f"{row['Margen_promedio'] * 0.05:.2f}%")
                st.metric("Unidades Vendidas", f"{row['Unidades_vendidas']:,}", delta=f"{int(row['Unidades_vendidas'] * 0.1):,}")

            with col2:
                monthly_sales = df_filtrado[df_filtrado['Producto'] == row['Producto']].groupby('Fecha').agg(Unidades_vendidas=('Unidades_vendidas', 'sum')).reset_index()

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(monthly_sales['Fecha'], monthly_sales['Unidades_vendidas'], label='Unidades Vendidas', color='blue')
                z = np.polyfit(monthly_sales.index, monthly_sales['Unidades_vendidas'], 1)
                p = np.poly1d(z)
                ax.plot(monthly_sales['Fecha'], p(monthly_sales.index), "r--", label="Tendencia")
                ax.set_title(f"Evolución de Ventas Mensual - {row['Producto']}")
                ax.set_xlabel("Fecha (Año-Mes)")
                ax.set_ylabel("Unidades Vendidas")
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Ha ocurrido un error al procesar el archivo: {e}")

uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")
if uploaded_file:
    procesar_archivo(uploaded_file)
else:
    mostrar_informacion_alumno()
