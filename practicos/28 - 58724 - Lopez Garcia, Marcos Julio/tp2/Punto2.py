def comprobar_parentesis(tokens):
    contador = 0
    
    for token in tokens:
        if token == '(':
            contador += 1
        elif token == ')':
            contador -= 1
        
        # Si el contador es negativo, hay un paréntesis de cierre antes de uno de apertura.
        if contador < 0:
            return False
    
    # Si al final el contador no es 0, hay más paréntesis de apertura que de cierre.
    return contador == 0

# Prueba
tokens = ['(', '1', '+', '23', '*', '34', '+', '(', '15', '+', '10', ')', ')']
print(comprobar_parentesis(tokens))  # Debería retornar True

tokens_invalidos = ['(', '1', '+', '23', '*', '34', '+', '15', '+', '10', ')', ')']
print(comprobar_parentesis(tokens_invalidos))  # Debería retornar False
