# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes
from Clasificador import ClasificadorAPriori

print "Practica 2 test de Vecinos próximos"

print "\nFichero de datos: example3.data"
dataset=Datos('./ConjuntosDatos/example3.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
laplace = False
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"


