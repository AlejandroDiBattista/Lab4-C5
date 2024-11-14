from fasthtml.common import *

app, rt = fast_app()

contador = 10

def Incrementador(cantidad):
    return P(
                A(
                    f'Incrementar {cantidad}',
                    hx_put=f'/incrementar/{cantidad}',
                    hx_target='#contador'
            )
        )
    
def Contador():
    color = 'blue' if contador >= 0 else 'red'
    return H3(
        f"Contador: {contador}",
        id='contador',
        style=f'color:{color};'
    )

@rt('/')
def home():
    return Titled(
        'Mi Contador',    
        Contador(),
        Incrementador(-1),
        Incrementador(-5),
        Incrementador(1),
        Incrementador(5),
    )

@rt('/incrementar/{cantidad}',['PUT'])
def incrementar(cantidad:int):
    global contador
    contador = contador + cantidad
    return Contador()

@rt('/reset', ['DELETE'])
def reset():
    global contador
    contador = 0
    return Redirect('/')

serve()