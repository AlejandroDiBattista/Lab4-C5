from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import List, Optional

# Definición del modelo de datos para Contacto
class Contacto(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    nombre: str = Field(max_length=50)
    apellido: str
    telefono: str

https://fastht.ml 

# Crear la base de datos SQLite
DATABASE_URL = "sqlite:///./contactos.db"
engine = create_engine(DATABASE_URL, echo=True)

# Crear la base de datos y las tablas
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

create_db_and_tables()

def agregar_contactos_por_defecto():
    with Session(engine) as session:
        contactos_existentes = session.exec(select(Contacto)).all()
        if not contactos_existentes:
            contactos_por_defecto = [
                Contacto(nombre="Juan", apellido="Perez", telefono="123456789"),
                Contacto(nombre="Maria", apellido="Gomez", telefono="987654321"),
                Contacto(nombre="Carlos", apellido="Lopez", telefono="555555555")
            ]
            session.add_all(contactos_por_defecto)
            session.commit()

agregar_contactos_por_defecto()

app = FastAPI()

# Dependencia para obtener una sesión de la base de datos
def get_session():
    with Session(engine) as session:
        yield session

# Endpoint para crear un contacto
@app.post("/contactos/", response_model=Contacto)
def crear_contacto(contacto: Contacto, session: Session = Depends(get_session)):
    session.add(contacto)
    session.commit()
    session.refresh(contacto)
    return contacto

# Endpoint para obtener todos los contactos
@app.get("/contactos/", response_model=List[Contacto])
def obtener_contactos(session: Session = Depends(get_session)):
    contactos = session.exec(select(Contacto)).all()
    return contactos

# Endpoint para obtener un contacto por ID
@app.get("/contactos/{contacto_id}", response_model=Contacto)
def obtener_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    return contacto

# Endpoint para actualizar un contacto por ID
@app.put("/contactos/{contacto_id}", response_model=Contacto)
def actualizar_contacto(contacto_id: int, contacto: Contacto, session: Session = Depends(get_session)):
    contacto_db = session.get(Contacto, contacto_id)
    if not contacto_db:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    contacto_db.nombre = contacto.nombre
    contacto_db.apellido = contacto.apellido
    contacto_db.telefono = contacto.telefono
    session.commit()
    session.refresh(contacto_db)
    return contacto_db

# Endpoint para eliminar un contacto por ID
@app.delete("/contactos/{contacto_id}", response_model=dict)
def eliminar_contacto(contacto_id: int, session: Session = Depends(get_session)):
    contacto = session.get(Contacto, contacto_id)
    if not contacto:
        raise HTTPException(status_code=404, detail="Contacto no encontrado")
    session.delete(contacto)
    session.commit()
    return {"message": "Contacto eliminado correctamente"}

# Crear la base de datos al iniciar la aplicación
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

n = "); DELETE FROM contacto; --"
f"INSERT INTO contacto (nombre, apellido, telefono) VALUES ({n}, 'Pérez', '123456789');
