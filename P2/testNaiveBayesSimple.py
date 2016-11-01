# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorNaiveBayes

print "Prueba 2- Fichero de datos d3.data"
dataset=Datos('./ConjuntosDatosPruebas/d3.data',True)
print "Estrategia: validacion simple, numParticiones: 1 porcentaje 80"
estrategia=ValidacionSimple(1,80)
#print estrategia.nombreEstrategia
print "Clasificador: clasificador Naive Bayes"
clasificador = ClasificadorNaiveBayes()
print "Errores: "
correccionL = True #aplicar correcion de Laplace a la tabla de train o no
errores=clasificador.validacion(estrategia,dataset,clasificador,correccionL)


