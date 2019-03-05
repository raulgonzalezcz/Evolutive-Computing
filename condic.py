booleana = True
numero = 6.1
cadena = "adios"
calificaciones = [9.5,10,7.3,6.4,9]

print("Comparacion de cadenas")
# Cadena es igual a "hola"?
if cadena == "hola":
	print("Saludo")
else:
	print("Despedida")

# Cadena es distinta a "hola"?
if not(cadena == "hola"):
	print("Despedida")
else:
	print("Saludo")

print("Comparacion de numeros usando elif. Si se cumple una condición, los demás no se usan")
# elif = else + if
promedio = sum(calificaciones) / len(calificaciones)
print(promedio)
if promedio > 6: #[0,5.9999]
	print("Reprobados")
elif promedio > 7: #[6,6.99999]
	print("R")
elif promedio > 8: #[7,7.999]
	print("B")
elif promedio > 9: #[8,8.99]
	print("MB")
else:
	print("E")

print("Comparacion de números usando if simple. Evalua todas las condiciones")
if promedio > 6: #[0,5.9999]
	print("Reprobados")
if promedio > 7: #[6,6.99999]
	print("R")
if promedio > 8: #[7,7.999]
	print("B")
if promedio > 9: #[8,8.99]
	print("MB")
else:
	print("E")

print("Evalua si un elemento está en una lista")
if 8 in calificaciones:
	print("Encontrado")
if not( 8 in calificaciones):
	print("No está")