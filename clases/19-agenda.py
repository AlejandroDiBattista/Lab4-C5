from fasthtml.common import *

app, rt = fast_app()

@dataclass
class Persona:
    id:int
    nombre:str
    telefono:str

agenda = [
    Persona(1,'Juan', '1234'),
    Persona(2,'Pedro', '5678'),
    Persona(3,'Ana', '9876'),
]
def Agregar():
    return Form(
        H3('Agregar Contacto'),
        Hidden(name='id', value='0'),
        Input(placeholder='Nombre', name='nombre'),
        Input(placeholder='Teléfono', name='telefono'),
        Button('Agregar', hx_post='/agregar', hx_target='#agenda'),
        id='formulario'
    )

def Editar(contacto):
    return Form(
        H3('Editar Contacto'),
        Hidden(name='id', value=contacto.id),
        Input(placeholder='Nombre', name='nombre',value=contacto.nombre),
        Input(placeholder='Teléfono', name='telefono',value=contacto.telefono),
        Button('Agregar', hx_post='/agregar', hx_target='#agenda'),
        id='formulario'
    )

def Mostrar(contacto):
    return Li(
                H4(contacto.nombre),
                P(contacto.telefono),
                A('Eliminar', 
                  hx_delete=f'/eliminar/{contacto.id}', 
                  hx_target=f"#contacto-{contacto.id}"),
                  " | ",
                A('Editar', 
                  hx_get=f'/editar/{contacto.id}', 
                  hx_target=f"#formulario"),
                id=f"contacto-{contacto.id}"

            ) 
def MostrarAgenda():
    return Ol(
        *[Mostrar(p) for p in agenda],
        id='agenda'
    )

@rt('/')
def home():
    return Titled(
        'Mi Agenda',
        Agregar(),  
        MostrarAgenda(),
    )

@rt('/eliminar/{id}', ['DELETE'])
def eliminar(id:int):
    global agenda
    agenda = [p for p in agenda if p.id != id]

@rt('/agregar', ['POST'])
def agregar(contacto:Persona):
    global agenda
    contacto.id = len(agenda)+1
    agenda.append(contacto)
    return MostrarAgenda()

@rt('/editar/{id}', ['GET'])
def editar(id:int):
    global agenda
    contacto = [p for p in agenda if p.id == id][0]
    return Editar(contacto)

serve()