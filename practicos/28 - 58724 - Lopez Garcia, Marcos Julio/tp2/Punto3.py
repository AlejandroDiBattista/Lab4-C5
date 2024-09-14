def es_operador(token):
    return token in "+-*/"

def es_numero(token):
    return token.isdigit()

def comprobar_expresion_valida(tokens):
    if not tokens:
        return False

    # Consideramos que la expresión está correctamente balanceada en paréntesis
    # Vamos a recorrer la lista de tokens
    anterior = None
    for token in tokens:
        if es_numero(token):
            # Si el anterior era un número, no puede haber dos números seguidos sin operador
            if anterior and es_numero(anterior):
                return False
        elif es_operador(token):
            # No puede haber operadores seguidos
            if anterior is None or es_operador(anterior):
                return False
        elif token == '(':
            # Si encontramos un paréntesis de apertura, lo tratamos como válido
            if anterior and es_numero(anterior):
                return False  # No puede haber un número antes de un paréntesis abierto
        elif token == ')':
            # Si encontramos un paréntesis de cierre, debe haber una expresión válida antes
            if anterior is None or es_operador(anterior) or anterior == '(':
                return False
        anterior = token
    
    # El último token no puede ser un operador
    return not es_operador(anterior)

# Pruebas
tokens_validos = ['(', '1', '+', '23', '*', '34', ')', '+', '15']
print(comprobar_expresion_valida(tokens_validos))  # True

tokens_invalidos = ['(', '1', '+', '*', '34', ')']
print(comprobar_expresion_valida(tokens_invalidos))  # False
