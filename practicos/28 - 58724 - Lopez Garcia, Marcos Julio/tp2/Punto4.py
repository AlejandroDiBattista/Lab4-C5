def evaluar(tokens):
    # Paso 1: Resolver paréntesis
    while '(' in tokens:
        # Encontramos el paréntesis más interno
        inicio = None
        for i, token in enumerate(tokens):
            if token == '(':
                inicio = i
            elif token == ')' and inicio is not None:
                subexpresion = tokens[inicio + 1:i]
                # Evaluamos la subexpresión
                resultado = evaluar(subexpresion)
                # Reemplazamos la subexpresión completa por su resultado
                tokens = tokens[:inicio] + [str(resultado)] + tokens[i + 1:]
                break

    # Paso 2: Resolver multiplicaciones
    while '*' in tokens:
        for i, token in enumerate(tokens):
            if token == '*':
                # Multiplicamos el número anterior por el siguiente
                resultado = int(tokens[i - 1]) * int(tokens[i + 1])
                tokens = tokens[:i - 1] + [str(resultado)] + tokens[i + 2:]
                break

    # Paso 3: Resolver sumas
    resultado = int(tokens[0])
    i = 1
    while i < len(tokens):
        operador = tokens[i]
        numero = int(tokens[i + 1])
        if operador == '+':
            resultado += numero
        i += 2

    return resultado

# Prueba
tokens = ['(', '1', '+', '23', '*', '34', ')', '+', '15']
print(evaluar(tokens))  # Resultado esperado: 808
