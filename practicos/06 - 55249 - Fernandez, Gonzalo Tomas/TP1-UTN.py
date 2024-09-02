from datetime import date

# Fecha Actual
print(date.today())
# Alumno: Fernandez, Gonzalo Tomas(ProgramadorPromedio02)
student = 'Fernandez, Gonzalo Tomas'
print(f'Alumno: {student}')
# Legajo: 55249
file = '55249'
print(f'Legajo: {file}')
# Carrera y Comisión:
career_and_commission = 'TUP C5'
print(f'Carrera y Comisión: {career_and_commission}')

# Trabajo Práctico 1
print('Trabaja Práctico 1\n')

## Ejercicio 1:
print('Ejercicio 1:')

phrases = [
  "Python es un lenguaje de programación",
  "Es fuertemente tipado y de alto nivel",
  "Por eso me gusta resolver problemas con este lenguaje",
  "Las listas y los bucles son muy útiles",
  "Son muy importantes en la funciones"
]

def phrases_process(phrases):
  words_per_phrase = []
  characters_per_phrase = []
  for phrase in phrases:
    words = phrase.split()
    count_words = len(words)
    count_characters = len(phrase)

    words_per_phrase.append(count_words)
    characters_per_phrase.append(count_characters)
    
    print(f'La frase: "{phrase}"')
    print(f'Tiene {count_words} palabras y {count_characters} caracteres.')
    print()

phrases_process(phrases)

## Ejercicio 2:
print('Ejercicio 2:')

def find_values():
  for a in range(-10, 11):
    for b in range(-10, 11):
      for c in range(-10, 11):
        if (a*0**2 + b*0 + c == 0 and
            a*1**2 + b*1 + c == 8 and
            a*2**2 + b*2 + c == 12 and
            a*3**2 + b*3 + c == 12 and
            a*5**2 + b*5 + c == 0):
            return a, b, c
  return None

a, b, c = find_values()
print(f"Los valores son: a = {a}, b = {b}, c = {c}")
