# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes


print "Prueba 2- Fichero de datos tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
#print estrategia.nombreEstrategia
print "Clasificador: clasificador Naive Bayes"
clasificador = ClasificadorNaiveBayes()
print "Errores: "
errores=clasificador.validacion(estrategia,dataset,clasificador)