from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional
from fasthtml.common import *

# Configuración de la base de datos
DATABASE_URL = "sqlite:///contacts.db"
engine = create_engine(DATABASE_URL)

# Definición del modelo de contacto
class Contact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    phone: Optional[str] = None

# Crear la tabla de contactos si no existe
SQLModel.metadata.create_all(engine)

# Configuración de FastHTML
app, rt = fast_app()

# Ruta para ver todos los contactos
@rt("/")
def get_contacts():
    with Session(engine) as session:
        contacts = session.exec(select(Contact)).all()
    return Titled("Lista de Contactos", 
        Div(*[Div(P(f"ID: {c.id}, Nombre: {c.name}, Email: {c.email}, Teléfono: {c.phone}"), 
                Button("Editar", hx_get=f"/edit/{c.id}", cls="secondary"), 
                Button("Borrar", hx_post=f"/delete/{c.id}", cls="danger")) 
            for c in contacts],
            Div(P(A("Agregar Nuevo Contacto", href="/new"))))
    )

# Ruta para agregar un nuevo contacto
@rt("/new")
def new_contact():
    form = Form(method="post", action="/add")(
        Label("Nombre:", Input(name="name")),
        Label("Email:", Input(name="email")),
        Label("Teléfono:", Input(name="phone")),
        Button("Guardar", type="submit")
    )
    return Titled("Nuevo Contacto", form)

@rt("/add")
def post(contact_data: dict):
    with Session(engine) as session:
        contact = Contact(**contact_data)
        session.add(contact)
        session.commit()
    return RedirectResponse(url="/")

# Ruta para editar un contacto existente
@rt("/edit/{contact_id}")
def edit_contact(contact_id: int):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if not contact:
            return Titled("Error", P("Contacto no encontrado"))
        form = Form(method="post", action=f"/update/{contact_id}")(
            Label("Nombre:", Input(name="name", value=contact.name)),
            Label("Email:", Input(name="email", value=contact.email)),
            Label("Teléfono:", Input(name="phone", value=contact.phone)),
            Button("Guardar", type="submit")
        )
        return Titled("Editar Contacto", form)

@rt("/update/{contact_id}")
def post(contact_id: int, contact_data: dict):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if contact:
            contact.name = contact_data.get("name", contact.name)
            contact.email = contact_data.get("email", contact.email)
            contact.phone = contact_data.get("phone", contact.phone)
            session.commit()
    return RedirectResponse(url="/")

# Ruta para eliminar un contacto
@rt("/delete/{contact_id}")
def post(contact_id: int):
    with Session(engine) as session:
        contact = session.get(Contact, contact_id)
        if contact:
            session.delete(contact)
            session.commit()
    return RedirectResponse(url="/")

serve()
