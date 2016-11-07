# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorVecinosProximos

print "Practica 2 test apartado 2"


# Fichero 1 example3.data

print "\nFichero de datos: example3.data"
print "Laplace = False, normalizar = False"
laplace = False
normalizar = False

dataset=Datos('./ConjuntosDatos/example3.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "\n\nLaplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/example3.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


#Fichero 2 - example 4

print "\n"
print "***********************************************************************"
print "\n\nFichero de datos: example4.data"
print "Laplace = False, normalizar = False"
laplace = False
normalizar = False

dataset=Datos('./ConjuntosDatos/example4.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "\n\nLaplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/example4.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


#Fichero 3 - wine_proc

print "\n"
print "***********************************************************************"
print "\n\nFichero de datos: wine_proc.data"
print "Laplace = False, normalizar = False"
laplace = False
normalizar = False

dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "\n\nLaplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"




#Fichero 4 - wdbc

print "\n"
print "***********************************************************************"
print "\n\nFichero de datos: wdbc.data"
print "Laplace = False, normalizar = False"
laplace = False
normalizar = False

dataset=Datos('./ConjuntosDatos/wdbc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "\n\nLaplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/wdbc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"




#Fichero 5 - digits

print "\n"
print "***********************************************************************"
print "\n\nFichero de datos: digits.data"
print "Laplace = False, normalizar = False"
laplace = False
normalizar = False

dataset=Datos('./ConjuntosDatos/digits.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "\n\nLaplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/digits.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=3"
clasificador = ClasificadorVecinosProximos(3)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=5"
clasificador = ClasificadorVecinosProximos(5)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=11"
clasificador = ClasificadorVecinosProximos(11)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=21"
clasificador = ClasificadorVecinosProximos(21)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: clasificador a Vecinos próximos k=51"
clasificador = ClasificadorVecinosProximos(51)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"