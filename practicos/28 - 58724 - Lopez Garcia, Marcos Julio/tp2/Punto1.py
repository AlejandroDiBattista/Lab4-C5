def extraer_tokens(expresion):
    tokens = []
    numero = ""
    
    for char in expresion:
        if char.isdigit():
            # Si el carácter es un dígito, lo añadimos al número en construcción.
            numero += char
        else:
            if numero:
                # Si tenemos un número en construcción, lo añadimos a la lista de tokens.
                tokens.append(numero)
                numero = ""  # Reseteamos el número en construcción.
            
            if char in "()+-*/":
                # Si el carácter es un operador o un paréntesis, lo añadimos como un token.
                tokens.append(char)
            elif char.isspace():
                # Si es un espacio, lo ignoramos.
                continue
    
    # Si quedó algún número al final, lo añadimos a la lista de tokens.
    if numero:
        tokens.append(numero)
    
    return tokens

# Prueba de la función
expresion = "(1 + 23 * 34 + (15 + 10))"
resultado = extraer_tokens(expresion)
print(resultado)

