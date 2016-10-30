# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorVecinosProximos

print "Practica 2 test de Vecinos próximos"

print "\nFichero de datos: example3.data"
dataset=Datos('./ConjuntosDatos/example3.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Vecinos próximos k=1"
clasificador = ClasificadorVecinosProximos(1)
print "Ejecucción: "
laplace = False
normalizar = True
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