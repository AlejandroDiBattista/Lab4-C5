# Revisión de Código y Errores Detectados

## Resumen
El código presenta **4 errores significativos** que impiden que las pruebas pasen correctamente. La implementación actual requiere ajustes específicos para cumplir con los requisitos de las pruebas.

## Detalles de los Errores

### 1. Constructor de la Clase `Factura` Incorrectamente Definido
- **Descripción del Error**: El constructor de la clase `Factura` está definido como `_init_` con un solo guión bajo, lo que impide que Python lo reconozca como el constructor adecuado.
- **Cambio Necesario**: Renombrar el método `_init_` a `__init__` (con dos guiones bajos antes y después).
- **Indicaciones para la Corrección**:
  ```python
  # Cambiar esto:
  def _init_(self, catalogo, cliente):
  
  # Por esto:
  def __init__(self, catalogo, cliente):
  ```

### 2. Método `agregar` de la Clase `Catalogo` No Acepta Múltiples Productos
- **Descripción del Error**: En las pruebas, el método `agregar` de la clase `Catalogo` se invoca con múltiples argumentos (productos), pero su implementación actual solo acepta un único producto.
- **Cambio Necesario**: Modificar el método `agregar` para que pueda recibir múltiples productos mediante argumentos variables.
- **Indicaciones para la Corrección**:
  ```python
  # Cambiar la definición del método:
  def agregar(self, producto):
      self.productos.append(producto)
  
  # A una que acepte múltiples productos:
  def agregar(self, *productos):
      for producto in productos:
          self.productos.append(producto)
  ```

### 3. Método `calcularTotales` de la Clase `Factura` Invoca Incorrectamente `calcularDescuentos`
- **Descripción del Error**: Dentro de `calcularTotales`, se llama a `self.catalogo.calcularDescuentos(self.productos)`, pero la clase `Catalogo` solo tiene el método `calcularDescuento(producto, cantidad)`.
- **Cambio Necesario**: Ajustar la llamada para calcular descuentos por cada producto individualmente.
- **Indicaciones para la Corrección**:
  ```python
  # Cambiar esto dentro de calcularTotales:
  self.descuentos = self.catalogo.calcularDescuentos(self.productos)
  
  # Por algo como esto:
  self.descuentos = sum(
      self.catalogo.calcularDescuento(producto, cantidad)
      for producto, cantidad in self.productos.items()
  )
  ```

### 4. Método `imprimir` de la Clase `Factura` No Incluye Descripciones de Ofertas
- **Descripción del Error**: Las pruebas esperan que el método `imprimir` incluya descripciones como "Descuento 10%" y "Oferta 2x1", pero actualmente solo muestra los montos de descuentos y totales.
- **Cambio Necesario**: Incluir detalles de las ofertas aplicadas en la impresión de la factura.
- **Indicaciones para la Corrección**:
  ```python
  # Modificar el método imprimir para incluir descripciones de ofertas:
  def imprimir(self):
      impresion = f"Factura {self.numero}\nCliente: {self.cliente.nombre}\n"
      for producto, cantidad in self.productos.items():
        impresion += f"{cantidad}u {producto.nombre} ${producto.precio} = ${producto.precio * producto.cantidad}\n"
        oferta = self.catalogo.buscarOferta(producto, cantidad)
        if not oferta: continue
        descuento = oferta.calcularDescuento(producto, cantidad)
        impresion += f"{oferta.descripcion} = -${descuento}\n"
      
      impresion += f"Total: {self.total}\n"
      return impresion
  ```

## Puntaje Final
**6/10**

Cada error significativo ha reducido el puntaje en un punto, comenzando desde 10.
