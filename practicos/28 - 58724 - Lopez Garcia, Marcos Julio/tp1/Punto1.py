def procesar_frases(frases):
    # Lista para almacenar la cantidad de palabras y la cantidad de caracteres por frase
    cantidades_palabras = []
    cantidades_caracteres = []
    
    # Recorrer cada frase
    for frase in frases:
        # Convertir la frase en una lista de palabras
        palabras = frase.split()  # Divide la frase por espacios
        
        # Contar la cantidad de palabras y la cantidad de caracteres
        cantidad_palabras = len(palabras)
        cantidad_caracteres = len(frase)  # Incluye espacios en el conteo
        
        # Agregar los resultados a las listas
        cantidades_palabras.append(cantidad_palabras)
        cantidades_caracteres.append(cantidad_caracteres)
        
        # Imprimir la frase original junto con la cantidad de palabras
        print(f"Frase: '{frase}' - Palabras: {cantidad_palabras}")
    
    return cantidades_palabras, cantidades_caracteres

# Ejemplo de uso
frases = [
    "Python es un lenguaje de programación",
    "Me gusta resolver problemas con código",
    "Las listas y los bucles son muy útiles"
]

procesar_frases(frases)
