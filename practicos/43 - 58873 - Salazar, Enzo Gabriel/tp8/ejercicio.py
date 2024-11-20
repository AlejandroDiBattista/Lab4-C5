import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def mostrar_informacion_alumno():
    with st.container():
        st.markdown('**Legajo:** 58873')
        st.markdown('**Nombre:** Salazar Enzo Gabriel')
        st.markdown('**Comisión:** C5')


mostrar_informacion_alumno()


st.title("Análisis de Ventas por Producto")


opcion = st.selectbox("Selecciona el tipo de archivo", ["Gaseosas", "Vinos"])


if opcion == "Gaseosas":
    
    uploaded_file = st.file_uploader("Sube el archivo CSV de Gaseosas", type="csv")
else:
   
    uploaded_file = st.file_uploader("Sube el archivo CSV de Vinos", type="csv")


if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file)
    
    
    st.write("Primeras filas del archivo cargado:")
    st.write(df.head())
    
    
    sucursal = st.selectbox("Selecciona la sucursal", ["Todas"] + df['Sucursal'].unique().tolist())
    
    if sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]
    
    
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    

    grouped = df.groupby('Producto').agg(
        Unidades_vendidas=('Unidades_vendidas', 'sum'),
        Precio_promedio=('Precio_promedio', 'mean'),
        Margen_promedio=('Margen_promedio', 'mean')
    ).reset_index()
    
    
    st.subheader("Métricas por Producto")
    st.write(grouped)
    
  
    st.subheader("Evolución de las ventas por mes")
    
 
    df['Año'] = pd.to_numeric(df['Año'], errors='coerce') 
    df['Mes'] = pd.to_numeric(df['Mes'], errors='coerce')  
    
    
    df = df.dropna(subset=['Año', 'Mes'])
    
    
    st.write("Tabla 'Año' y 'Mes'")
    st.write(df[['Año', 'Mes']].head())
    
    
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2) + '-01', errors='coerce')
    
   
    st.write("control de fechas")
    st.write(df[['Año', 'Mes', 'Fecha']].head())
    
    
    monthly_sales = df.groupby('Fecha').agg(Unidades_vendidas=('Unidades_vendidas', 'sum')).reset_index()

    
    min_fecha = monthly_sales['Fecha'].min()  
    monthly_sales['Fecha_num'] = (monthly_sales['Fecha'] - min_fecha).dt.days  
    
   
    st.write("Tabla fecha de ventas")
    st.write(monthly_sales[['Fecha', 'Fecha_num', 'Unidades_vendidas']].head())
    
   
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_sales, x='Fecha_num', y='Unidades_vendidas', marker='o', label='Unidades Vendidas')
    plt.title('Evolución de Ventas por Mes')
    plt.xlabel('Fecha')
    plt.ylabel('Unidades Vendidas')
    plt.xticks(rotation=45)
    
   
    sns.regplot(data=monthly_sales, x='Fecha_num', y='Unidades_vendidas', scatter=False, color='red')
    
    st.pyplot(plt)

mostrar_informacion_alumno()