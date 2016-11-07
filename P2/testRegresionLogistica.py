# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorRegresionLogistica

print "Practica 2 test apartado 3"
nEpocas = 1
cteAprendizaje = 1

#Fichero 1 - d4.data
print "\nFichero de datos: d4.data"


print "Laplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatosPruebas/d4.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(5)
print "Clasificador: clasificador regresion logistica con nEpocas: ",nEpocas," y cte. de Aprendizaje" , cteAprendizaje
clasificador = ClasificadorRegresionLogistica(nEpocas, cteAprendizaje)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


