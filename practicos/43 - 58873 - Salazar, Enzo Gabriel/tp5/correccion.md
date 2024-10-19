# Revisión del Código de Facturación

Después de revisar el código proporcionado, se han detectado **2 errores significativos** que impiden que las pruebas se superen correctamente. A continuación, se detallan cada uno de estos errores junto con las indicaciones para corregirlos.

## Errores Encontrados

1. **Falta del Campo `fecha` y Formato Incorrecto en la Clase `Factura`**
2. **Formato Incorrecto en el Método `informe` de la Clase `Catalogo`**

---

### 1. Falta del Campo `fecha` y Formato Incorrecto en la Clase `Factura`

**Descripción del Error:**
La clase `Factura` no incluye el campo `fecha`, el cual es obligatorio según el enunciado. Además, el método `imprimir` no genera el formato de factura requerido, que incluye detalles como la fecha y la información específica de cada producto y sus descuentos.

**Indicaciones para Corregir:**

- **Agregar el Campo `fecha`:**
  Incluir un atributo `fecha` en el constructor de la clase `Factura`, asignándolo con la fecha actual al momento de crear la factura.

- **Actualizar el Método `imprimir`:**
  Modificar el método `imprimir` para que siga el formato especificado en el enunciado, incluyendo la fecha y detalles de cada producto con sus respectivos descuentos.

**Cambio Específico:**

- **Agregar el Campo `fecha` en el Constructor:**
  ```python
  from datetime import datetime

  class Factura:
      # ... código existente ...

      def __init__(self, catalogo, cliente):
          self.catalogo = catalogo
          self.cliente = cliente
          self.numero = Factura.nuevoNumero()
          self.fecha = datetime.now().strftime("%Y-%m-%d")  # Formato de fecha YYYY-MM-DD
          self.items = []
          self.descuentos = 0.0
  ```

- **Actualizar el Método `imprimir`:**
  ```python
  def imprimir(self):
      impresion = f"Factura: {self.numero}\n"
      impresion += f"Fecha  : {self.fecha}\n"
      impresion += f"Cliente: {self.cliente.nombre} ({self.cliente.cuit})\n\n"
      
      for producto, cantidad in self.items:
          subtotal = producto.precio * cantidad
          descuento = self.catalogo.calcularDescuento(producto, cantidad)
          impresion += f"- {cantidad}u {producto.nombre} x ${producto.precio} = ${subtotal}\n"
          if descuento > 0:
              impresion += f"      {self.catalogo.buscarOferta(producto).descripcion} - ${descuento}\n"
      
      impresion += f"\nSubtotal:   ${self.subtotal:.2f}\n"
      impresion += f"Descuentos: ${self.descuentos:.2f}\n"
      impresion += "-----------------------\n"
      impresion += f"Total:      ${self.total:.2f}\n"
      return impresion
  ```

---

### 2. Formato Incorrecto en el Método `informe` de la Clase `Catalogo`

**Descripción del Error:**
El método `informe` de la clase `Catalogo` no sigue el formato especificado en el enunciado, especialmente en la sección de "Tipos de productos", donde se requiere mostrar la cantidad de unidades y el precio promedio por tipo.

**Indicaciones para Corregir:**
Modificar el método `informe` para que presente la información de "Tipos de productos" en el formato requerido, incluyendo la cantidad de unidades y el precio promedio por tipo.

**Cambio Específico:**

- **Actualizar la Sección de "Tipos de productos" en el Método `informe`:**
  ```python
  def informe(self):
      informe = "INFORME CATALOGO\n"
      informe += f"Cantidad de productos:   {self.cantidadProductos}\n"
      informe += f"Cantidad de unidades:    {self.cantidadUnidades}\n"
      informe += f"Precio promedio:       $ {self.precioPromedio:.2f}\n"
      informe += f"Valor total:           $ {self.valorTotal:.2f}\n"
      informe += "Tipos de productos:\n"
      
      tipos = self.tiposProductos()
      for tipo in tipos:
          unidades = sum(p.cantidad for p in self.productos if p.tipo == tipo)
          precios = [p.precio for p in self.productos if p.tipo == tipo]
          precio_promedio_tipo = sum(precios) / len(precios) if precios else 0
          informe += f"  - {tipo}: {unidades}u x $ {precio_promedio_tipo:.2f}\n"
      
      informe += "Ofertas:\n"
      if self.ofertas:
          for oferta in self.ofertas:
              informe += f"  - {oferta.descripcion}\n"
      else:
          informe += "  - No hay ofertas disponibles.\n"
      
      return informe
  ```

---

## Puntaje Final

**Puntaje Inicial:** 10  
**Puntos Restados por Errores:** 32  
****Puntaje Final:** 8/10**
