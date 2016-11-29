# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import AlgoritmoGenetico

print "Practica 3 test AG"
tamPoblacion = 5 #Tamaño de la poblacion
numGeneraciones = 3 #Numero de generaciones (Condicion de terminacion)
maxReglas = 3 #Numero máximo de reglas por individuo


#Fichero 1 - d4.data
print "\nFichero de datos: tic-tac-toe.data"


print "Laplace = False, normalizar = True"
laplace = False
normalizar = True

numParticionesSimples = 1 
porcentajeParticiones = 80

dataset=Datos('./ConjuntosDatos/pruebas_short_priori_B.data',True)
print "Estrategia: validacion simple, numParticiones: ",numParticionesSimples,", porcentaje: ", porcentajeParticiones
estrategia=ValidacionSimple(numParticionesSimples, porcentajeParticiones)
print "Clasificador: clasificador Algoritmo Genético tamPoblacion: ",tamPoblacion," y numGeneraciones" , numGeneraciones
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


