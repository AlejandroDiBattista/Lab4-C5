from fasthtml.common import *

app, rt = fast_app()

@rt('/')
def home():
    return Titled(
        'Hola Mundo',    
        P('Este es un párrafo de prueba'),
    )


serve()

# <div id="contacto" class="destacado">
#     <h1 style="color:red;">Contacto</h1>
#     <p>Correo: </p>
# </div>

# Div(
#     H1('Contacto', style='color:red;'),
#     P('Correo: '),
#     id='contacto',
#     cls='destacado'
# )