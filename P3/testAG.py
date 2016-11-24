# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import AlgoritmoGenetico

print "Practica 3 test AG"
tamPoblacion = 10 #Tamaño de la poblacion
numGeneraciones = 100 #Numero de generaciones (Condicion de terminacion)

#Fichero 1 - d4.data
print "\nFichero de datos: tic-tac-toe.data"


print "Laplace = False, normalizar = True"
laplace = False
normalizar = True

dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(5)
print "Clasificador: clasificador Algoritmo Genético tamPoblacion: ",tamPoblacion," y numGeneraciones" , numGeneraciones
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


