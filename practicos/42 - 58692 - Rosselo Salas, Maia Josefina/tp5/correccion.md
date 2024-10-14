# Revisión de Código: Sistema de Facturación

Tras revisar el código proporcionado, se han detectado **2 errores significativos** que impiden que las pruebas se ejecuten correctamente.

## Resumen de Errores

1. **Cálculo Incorrecto de Subtotal y Descuentos en la Clase `Factura`**
2. **Formato de Impresión de la Factura no Cumple con el Enunciado**

---

### 1. Cálculo Incorrecto de Subtotal y Descuentos en la Clase `Factura`

**Descripción del Error:**

El cálculo del subtotal y los descuentos en la clase `Factura` no coincide con lo esperado en las pruebas. Específicamente:

- **Subtotal:** La suma de los precios de los productos multiplicados por la cantidad vendida no coincide con los valores esperados en las pruebas.
- **Descuentos:** La suma de los descuentos aplicados no refleja correctamente las ofertas registradas, lo que resulta en discrepancias con los valores esperados.

**Indicaciones para Corregir el Código:**

- **Ajustar el Método `_actualizar_totales`:** Asegurarse de que los descuentos se calculen correctamente según las ofertas aplicables y que el subtotal refleje la suma total antes de descuentos.

**Cambios Puntuales a Realizar:**

```python
def _actualizar_totales(self):
    self._subtotal = 0
    self._descuentos = 0
    self.ofertas_aplicadas = []

    for producto, cantidad in self.articulos:
        subtotal_producto = producto.precio * cantidad
        self._subtotal += subtotal_producto

        descuento = self.catalogo.calcularDescuento(producto, cantidad)
        self._descuentos += descuento

        if cantidad > 1 and self.catalogo.tieneOferta2x1(producto):
            self.ofertas_aplicadas.append("Oferta 2x1")

        if descuento > 0:
            porcentaje_descuento = (descuento / subtotal_producto) * 100
            if porcentaje_descuento == 10:
                self.ofertas_aplicadas.append("Descuento 10%")
            # Agregar manejo para otros porcentajes si es necesario
```

**Instrucciones Específicas:**

- **Verificar la Aplicación de Descuentos:** Asegurarse de que si una oferta es aplicable, el descuento se calcula y se acumula correctamente.
- **Actualizar `ofertas_aplicadas`:** Añadir descripciones de ofertas solo cuando se aplican correctamente los descuentos correspondientes.

---

### 2. Formato de Impresión de la Factura no Cumple con el Enunciado

**Descripción del Error:**

El método `imprimir` de la clase `Factura` no genera el texto en el formato especificado en el enunciado. Falta detallar cada línea de producto con cantidad, precio y subtotales, así como los descuentos aplicados de manera detallada.

**Indicaciones para Corregir el Código:**

- **Modificar el Método `imprimir`:** Asegurarse de que el texto generado siga el formato exacto solicitado, incluyendo detalles de cada producto y los descuentos aplicados.

**Cambios Puntuales a Realizar:**

```python
def imprimir(self):
    impresion = f"Factura: {self.numero}\n"
    impresion += f"Fecha  : {datetime.now().strftime('%Y-%m-%d')}\n"
    impresion += f"Cliente: {self.cliente.nombre} ({self.cliente.cuit})\n\n"

    for producto, cantidad in self.articulos:
        subtotal_producto = producto.precio * cantidad
        descuento = self.catalogo.calcularDescuento(producto, cantidad)
        impresion += f"- {cantidad}u {producto.nombre:<20} x ${producto.precio} = ${subtotal_producto}\n"
        if descuento > 0:
            impresion += f"      {self.ofertas_aplicadas.pop(0):<30} - ${descuento}\n"

    impresion += f"\n{'Subtotal:':>50} ${self.subtotal:.2f}\n"
    impresion += f"{'Descuentos:':>50} -${self.descuentos:.2f}\n"
    impresion += f"{'-'*55}\n"
    impresion += f"{'Total:':>50} ${self.total:.2f}\n"
    return impresion
```

**Instrucciones Específicas:**

- **Formato de Productos:** Cada línea de producto debe mostrar la cantidad, nombre del producto, precio unitario y subtotal.
- **Descuentos Aplicados:** Deben mostrarse inmediatamente después de cada producto si aplican, con la descripción de la oferta y el monto descontado.
- **Totales:** Incluir el subtotal general, total de descuentos y el total final alineados correctamente.

---

## Puntaje Final: 8/10

- **Inicial:** 10 puntos
- **Errores Significativos:** 2
- **Puntaje Final:** 10 - 2 = **8**

Se recomienda realizar las correcciones mencionadas para asegurar que todas las pruebas se ejecuten correctamente y que la implementación cumpla con los requerimientos especificados en el enunciado.