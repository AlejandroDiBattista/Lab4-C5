def encontrar_coeficientes():
    # Valores de x y y proporcionados
    puntos = [(0, 0), (1, 8), (2, 12), (3, 12), (5, 0)]

    # Explorar posibles valores de a, b y c
    for a in range(-10, 11):  # a en el rango de [-10, 10]
        for b in range(-10, 11):  # b en el rango de [-10, 10]
            for c in range(-10, 11):  # c en el rango de [-10, 10]
                es_solucion = True
                # Comprobar si los coeficientes funcionan para todos los puntos
                for x, y in puntos:
                    f_x = a * x**2 + b * x + c  # Evaluamos la función f(x)
                    if f_x != y:  # Si no coincide con el valor y del punto, no es la solución
                        es_solucion = False
                        break
                
                if es_solucion:
                    # Si la función pasa por todos los puntos, retornamos los valores de a, b, y c
                    return a, b, c

# Prueba de la función
a, b, c = encontrar_coeficientes()
print(f"Los valores de a, b y c que hacen que la función pase por los puntos son: a={a}, b={b}, c={c}")
