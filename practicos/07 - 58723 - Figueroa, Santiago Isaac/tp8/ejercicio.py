import streamlit as s
import pandas as p
import numpy as n
import matplotlib.pyplot as pl
from functools import reduce
import itertools

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
url = 'https://proyectofinallabt8santiagofigueroa.streamlit.app/'

def mostrar_informacion_alumno():
    with s.container(border=True):
        s.markdown('**Legajo:** 58.723')
        s.markdown('**Nombre:** Santiago Figueroa')
        s.markdown('**Comisión:** C5')

class VentasEstrategia:
    def __init__(self, inform4ci0n):
        self.datos = inform4ci0n
        self.datos_transformados = self._transformar_datos()

    def _transformar_datos(self):
        return {
            pr0dúct0_: self.datos[self.datos['Producto'] == pr0dúct0_] 
            for pr0dúct0_ in self.datos['Producto'].unique()
        }

    def calcular_precio_promedio(self, dt4pr0duct__):
        return reduce(
            lambda acc, x: acc + x, 
            dt4pr0duct__['Ingreso_total'] / dt4pr0duct__['Unidades_vendidas']
        ) / len(dt4pr0duct__)

    def calcular_variacion_precio(self, dt4pr0duct__):
        precios_por_año = dt4pr0duct__.groupby('Año').apply(
            lambda x: n.mean(x['Ingreso_total'] / x['Unidades_vendidas'])
        )
        return n.mean(n.diff(precios_por_año) / precios_por_año[:-1]) * 100

    def calcular_margen(self, dt4pr0duct__):
        ingresos = dt4pr0duct__['Ingreso_total']
        costos = dt4pr0duct__['Costo_total']
        ganancias = list(map(lambda ing, cos: ing - cos, ingresos, costos))
        margenes = list(map(lambda gan, ing: (gan / ing) * 100, ganancias, ingresos))
        return n.mean(margenes)

    def calcular_variacion_margen(self, dt4pr0duct__):
        ingresos = dt4pr0duct__['Ingreso_total']
        costos = dt4pr0duct__['Costo_total']
        ganancias = list(map(lambda ing, cos: ing - cos, ingresos, costos))
        margenes_por_año = dt4pr0duct__.groupby('Año').apply(
            lambda x: n.mean([(gan / ing) * 100 for gan, ing in zip(
                x['Ingreso_total'] - x['Costo_total'], 
                x['Ingreso_total']
            )])
        )
        return n.mean(n.diff(margenes_por_año) / margenes_por_año[:-1]) * 100

    def calcular_unidades_totales(self, dt4pr0duct__):
        return n.sum(dt4pr0duct__['Unidades_vendidas'])

    def calcular_variacion_unidades(self, dt4pr0duct__):
        unidades_por_año = dt4pr0duct__.groupby('Año')['Unidades_vendidas'].sum()
        unidades_consecutivas = list(itertools.pairwise(unidades_por_año))
        variaciones = [((segundo - primero) / primero) * 100 for primero, segundo in unidades_consecutivas]
        return n.mean(variaciones)

def grafico(dt4pr0duct__, pr0dúct0_):
    vent4s_ = dt4pr0duct__.groupby(['Año', 'Mes'])['Unidades_vendidas'].sum().reset_index()
    x = n.arange(len(vent4s_))
    y = vent4s_['Unidades_vendidas']
    
    i, s = pl.subplots(figsize=(12, 7), dpi=200, facecolor='#f0f0f0')
    
    s.plot(x, y, label=pr0dúct0_, color='purple', marker='o', 
            linewidth=3, markersize=8, 
            markerfacecolor='purple', markeredgecolor='purple')
    
    z = n.polyfit(x, y, 1)
    p = n.poly1d(z)
    s.plot(x, p(x), linestyle='--', color='darkred', 
            label='Tendencia', linewidth=3)
    
    s.set_xticks(x)
    s.set_xticklabels([f"{row.Año}" if row.Mes == 1 else "" for row in vent4s_.itertuples()])
    
    s.set_title("Evolución de Ventas", 
                 fontsize=10, color='#333333')
    s.set_xlabel("Año-Mes", fontsize=13, color='#555555')
    s.set_ylabel("Unidades Vendidas", fontsize=12, color='#555555')
    
    s.grid(linestyle='--', alpha=0.6, color='#cccccc')
    
    s.legend(loc='best', frameon=True, 
              facecolor='white', edgecolor='lightgray')
    
    for spine in s.spines.values():
        spine.set_edgecolor('#cccccc')

    return i

def main():
    s.sidebar.header("Cargar archivos de ventas")
    csv4rch__ = s.sidebar.file_uploader("Cargar Archivo", type="csv")

    if csv4rch__:
        inform4ci0n = p.read_csv(csv4rch__)
        c4lcul0_ = VentasEstrategia(inform4ci0n)
        
        tiénd4s_ = ["Total"] + list(inform4ci0n['Sucursal'].unique())
        tiénd4 = s.sidebar.selectbox("Seleccionar Sucursal", tiénd4s_)
        
        if tiénd4 != "Total":
            inform4ci0n = inform4ci0n[inform4ci0n['Sucursal'] == tiénd4]
            s.title(f"Panel de Análisis: {tiénd4}")
        else:
            s.title("Datos de todas las Sucursales")

        próduct_0 = inform4ci0n['Producto'].unique()
        for pr0dúct0_ in próduct_0:
            s.subheader(f"Datos de: {pr0dúct0_}")
            dt4pr0duct__ = inform4ci0n[inform4ci0n['Producto'] == pr0dúct0_]
            pr0medi0c4lcul0 = c4lcul0_.calcular_precio_promedio(dt4pr0duct__)
            vari4ci0nc4lcul0_ = c4lcul0_.calcular_variacion_precio(dt4pr0duct__)
            m4rgenc4lcul0_ = c4lcul0_.calcular_margen(dt4pr0duct__)
            m4rgenv_ = c4lcul0_.calcular_variacion_margen(dt4pr0duct__)
            ut0t4les_ = c4lcul0_.calcular_unidades_totales(dt4pr0duct__)
            unid4desv_ = c4lcul0_.calcular_variacion_unidades(dt4pr0duct__)

            metric4col1 , metric4col2 = s.columns([1, 3])
            with metric4col1 :
                s.metric("Precio Promedio", f"${pr0medi0c4lcul0:,.0f}", f"{vari4ci0nc4lcul0_:.2f}%")
                s.metric("Margen Promedio", f"{m4rgenc4lcul0_:.0f}%", f"{m4rgenv_:.2f}%")
                s.metric("Unidades Vendidas", f"{ut0t4les_:,}", f"{unid4desv_:.2f}%")

            with metric4col2:
                i = grafico(dt4pr0duct__, pr0dúct0_)
                s.pyplot(i)


main()

mostrar_informacion_alumno()

