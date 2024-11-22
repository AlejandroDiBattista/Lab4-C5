import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

url = 'https://laboratoriotp8-gjzwyvcnmvavwexpo83aag.streamlit.app/'
def mostrar_informacion_alumno():
    st.markdown('**Legajo:** 58873')
    st.markdown('**Nombre:** Salazar Enzo Gabriel')
    st.markdown('**Comisión:** C5')


st.set_page_config(layout="wide")
st.title("Análisis de Ventas por Producto")


st.sidebar.header("Cargar archivo de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")
sucursal = st.sidebar.selectbox("Seleccionar Sucursal", ["Todas"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    #
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce')  
    df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')  
    df = df.dropna(subset=['Año', 'Mes'])  
    df = df[(df['Año'] >= 2000) & (df['Año'] <= 2024)]  
    df['Mes'] = df['Mes'].apply(lambda x: x if 1 <= x <= 12 else None) 
    df = df.dropna(subset=['Mes'])  

    
    df['Fecha'] = pd.to_datetime(
        df['Año'].astype(int).astype(str) + '-' +
        df['Mes'].astype(int).astype(str).apply(lambda x: x.zfill(2))  
    )

    
    if sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]

    
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']

    grouped = df.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()

    
    st.header("Datos de Todas las Sucursales")
    for i, row in grouped.iterrows():
        col1, col2 = st.columns([1, 3])

        
        with col1:
            st.subheader(row['Producto'])

            
            delta_precio = row['Precio_promedio'] * 0.1  
            delta_margen = row['Margen_promedio'] * 0.05  
            delta_unidades = row['Unidades_vendidas'] * 0.1  

            st.metric("Precio Promedio", f"${row['Precio_promedio']:.2f}",
                      delta=f"{delta_precio:.2f}", delta_color="normal")
            st.metric("Margen Promedio", f"{row['Margen_promedio']*100:.2f}%",
                      delta=f"{delta_margen:.2f}%", delta_color="inverse")
            st.metric("Unidades Vendidas", f"{row['Unidades_vendidas']:,}",
                      delta=f"{int(delta_unidades):,}", delta_color="normal")

        
        with col2:
            product_data = df[df['Producto'] == row['Producto']]

            
            monthly_sales = product_data.groupby('Fecha').agg(
                Unidades_vendidas=('Unidades_vendidas', 'sum')
            ).reset_index()

            
            plt.figure(figsize=(10, 6))
            plt.plot(
                monthly_sales['Fecha'], monthly_sales['Unidades_vendidas'],
                marker='o', linestyle='-', color='blue', label=row['Producto']
            )

            
            sns.regplot(
                x=monthly_sales.index,
                y=monthly_sales['Unidades_vendidas'],
                scatter=False,
                color='red',
                label='Tendencia',
                line_kws={"linewidth": 2}
            )

           
            ax = plt.gca()
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            ax.set_xlim([monthly_sales['Fecha'].min(), monthly_sales['Fecha'].max()])  

            plt.title(f"Evolución de Ventas Mensual - {row['Producto']}", fontsize=14)
            plt.xlabel("Fecha (Año-Mes)", fontsize=12)
            plt.ylabel("Unidades Vendidas", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()

            st.pyplot(plt)


mostrar_informacion_alumno()