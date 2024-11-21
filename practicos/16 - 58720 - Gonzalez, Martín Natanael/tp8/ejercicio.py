import streamlit
import pandas
import numpy
import matplotlib.pyplot
from matplotlib.dates import YearLocator, DateFormatter

#url = "https://segundoparciallaboratorio-gonzalezmartin.streamlit.app/"

streamlit.set_page_config(page_title="Segundó Parcial", layout="wide")

def mostreamlitrar_informacion_alumno():
    with streamlit.container(border=True):
        streamlit.markdown('**Legajo:** 58.720')
        streamlit.markdown('**Nombre:** Gonzalez Martin Natanael')
        streamlit.markdown('**Comisión:** C5')

mostreamlitrar_informacion_alumno()

with streamlit.sidebar:
    streamlit.header("Sube tu archivo CSV")
    archivo = streamlit.file_uploader("Sube tu archivo CSV", type=['csv'], label_visibility="collapsed")
    
    if archivo is not None:
        file_size_kb = len(archivo.getvalue()) / 1024
        streamlit.text(f"{archivo.name}\n{file_size_kb:.1f}KB")
    
    streamlit.header("Seleccionar Sucursal")
    sucursales = ['Todas', 'Sucursal Norte', 'Sucursal Centro', 'Sucursal Sur']
    sucursal_seleccionada = streamlit.selectbox('', sucursales, label_visibility="collapsed")


def calcular_datos(data_frame, sucursal='Todas'):
    if sucursal != 'Todas':
        data_frame = data_frame[data_frame['Sucursal'] == sucursal]
    
    productos = data_frame['Producto'].unique()
    productos_completos = {}
    
    for producto in productos:
      
        datos_productó = data_frame[data_frame['Producto'] == producto]
        
        datos_productó['Precio_promedio'] = datos_productó['Ingreso_total'] / datos_productó['Unidades_vendidas']
        precio_promedio = datos_productó['Precio_promedio'].mean()
        
        datos_productó['Margen'] = ((datos_productó['Ingreso_total'] - datos_productó['Costo_total']) / 
                                  datos_productó['Ingreso_total'] * 100)
        margen_promedio = datos_productó['Margen'].mean()
        
        unidades_vendidas = datos_productó['Unidades_vendidas'].sum()
        
        datos_anual = datos_productó.groupby('Año').agg({
            'Precio_promedio': 'mean',
            'Margen': 'mean',
            'Unidades_vendidas': 'sum'
        }).reset_index()
        
        variacion_precio = datos_anual['Precio_promedio'].pct_change().mean() * 100
        variacion_margen = datos_anual['Margen'].pct_change().mean() * 100
        variacion_unidades = datos_anual['Unidades_vendidas'].pct_change().mean() * 100
               
        datos_mensuales = datos_productó.groupby(['Año', 'Mes']).agg({
            'Precio_promedio': 'mean',
            'Margen': 'mean',
            'Unidades_vendidas': 'sum'
        }).reset_index()
        datos_mensuales['Fecha'] = pandas.to_datetime(
            datos_mensuales.apply(lambda x: f"{int(x['Año'])}-{int(x['Mes'])}-01", axis=1)
        )
              
        x = numpy.arange(len(datos_mensuales))
        z = numpy.polyfit(x, datos_mensuales['Unidades_vendidas'], 1)
        p = numpy.poly1d(z)
        
        productos_completos[producto] = dict(
        precio_promedio=precio_promedio,
        variacion_precio=variacion_precio,
        margen_promedio=margen_promedio,
        variacion_margen=variacion_margen,
        unidades_vendidas=unidades_vendidas,
        variacion_unidades=variacion_unidades,
        datos_mensuales=datos_mensuales,
        trend=p(x)
        )

    
    return productos_completos

def hacer_grafico(data, producto):
    graf = matplotlib.pyplot.figure(figsize=(10, 5))
    ax = graf.add_axes([0.1, 0.15, 0.85, 0.65])
    ax.grid(True, linestyle='-', alpha=0.2, color='gray')

    
    datos_mensualés = data['datos_mensuales']
    ax.plot(datos_mensualés['Fecha'], datos_mensualés['Unidades_vendidas'], label=producto, color='#1f77b4')
    ax.plot(datos_mensualés['Fecha'], data['trend'], label='Tendencia', color='red', linestyle='--')

    ax.set_title('Evolución de Ventas Mensual', pad=20, y=1.2)
    ax.set_ylabel('Unidades vendidas', labelpad=10)
    ax.set_xlabel('Año-Mes', labelpad=10)
    ax.legend(title='Producto', bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=2, borderaxespad=0)
    ax.set_ylim(bottom=0)  
    ax.set_xlim(pandas.Timestamp('2019-12-01'), pandas.Timestamp('2024-12-31'))
    ax.xaxis.set_major_locator(YearLocator(1))  
    ax.xaxis.set_major_formatter(DateFormatter('%Y')) 

    return graf

if archivo is not None:
    if sucursal_seleccionada == 'Todas':
      streamlit.title("Datós de Todas las Sucursalés")
    else:
      streamlit.title(f"Datós de la {sucursal_seleccionada}")
     
    data_frame = pandas.read_csv(archivo)
    results = calcular_datos(data_frame, sucursal_seleccionada)
    
    for producto, metrics in results.items():
      with streamlit.container(border=True):
        col_datos, col_grafico = streamlit.columns([1, 2])
        
        with col_datos:
            streamlit.markdown(f"<h1 streamlityle='padding-bottom: 50px;'>{producto}</h1>", unsafe_allow_html=True)
            streamlit.metric("Precio Promedío", f"${metrics['precio_promedio']:,.0f}", f"{metrics['variacion_precio']:.2f}%")
            streamlit.metric("Margen Promedío", f"{metrics['margen_promedio']:,.0f}%", f"{metrics['variacion_margen']:.2f}%")
            streamlit.metric("Unidades Vendídas", f"{metrics['unidades_vendidas']:,}", f"{metrics['variacion_unidades']:.2f}%")
        
        with col_grafico:
            graf = hacer_grafico(metrics, producto)
            streamlit.pyplot(graf)
            matplotlib.pyplot.close(graf)

        