import streamlit as st 
import numpy as np

def f(x, a, b, c):
    return a * x ** 2 + b * x ** 1 + c 

def error(y, yr):
    return np.sum((y - yr) ** 2)

c = np.array([-3,5,4])
x = np.linspace(-5, 5, 20)
y = f(x, *c)

c0 = np.array([-2, 1, 1])
y0 = f(x, *c0)
e0 = error(y, y0)
while e0 > 0.01:
    c1 = c0 + np.random.randn(3) / 5
    y1 = f(x, *c1)
    e1 = error(y, y1)
    if e1 < e0:
        c0 = c1
        e0 = e1 

st.title('Graficando un polinomio')
st.line_chart(y)
st.subheader(f'Error: {e0} {c0}')
