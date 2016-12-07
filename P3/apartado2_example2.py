# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from EstrategiaParticionado import ValidacionCruzada
from EstrategiaParticionado import ValidacionSimple
from AlgoritmoGenetico import AlgoritmoGenetico
from Main import Main

###################################################################################
#[MODE] - Establece los parametros de ejecucion.
#Prints:
#   default     - sólo aquellos prints que se nos piden + resultados.
#   verbose     - prints de debug en todas las fases, máx. de información (debug).
#Diversidad:
#   default     - sin alteraciones en el rand del num. de reglas.
#   maxReglas-1 - genera 3 reglas para cada individuo salvo el último
#                 (test alta diversidad en reglas por indv.).
#Reglas_extra:
#   default     - se trunca el numero de reglas si no matchea entre padre,madre.
#   randSon     - asigna de forma random las reglas extras a los hijos.
#
#Resto:
#   random      - se seleccionan los 'fill' de manera aleatoria.
#   fitness     - se seleccionan los 'fill' en funcion del fitness
#
#CondicionTerminacion:
#   numero      - a partir de este porcentaje de acierto se termina el entrenamiento 
#   no          - el entreamiento finaliza cuando se han ejecutado todas las generaciones  
###################################################################################
#default para pruebas:
mode = {'Prints':'default', 'Diversidad':'default', 'ReglasExtra':'randSons', 'Resto':'random', 'CondicionTerminacion':'100', 'Threshold':90.0}

#test: maximiza el numero de reglas por individuo menos el ultimo:
#mode = {'Prints':'verbose', 'Diversidad':'maxReglas-1', 'Reglas_extra':'randSon'}


print "Practica 3 test AG"

#validacion cruzada
numParticiones = 3
estrategiaCruzada=ValidacionCruzada(numParticiones)

#validacion Simple
numParticiones = 3
porcentajeParticiones = 80
estrategiaSimple=ValidacionSimple(numParticiones, porcentajeParticiones)

#Tweaks
laplace = False
normalizar = True
separador = False


#======================================example2.data=====================================#
data_file = 'example2.data'
about = "clasificador Algoritmo Genetico"
clf_id = "GenAlg"

#Algoritmo Genetico
tamPoblacion = 10 #Tamaño de la poblacion
numGeneraciones = 100 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)


#Algoritmo Genetico
tamPoblacion = 10 #Tamaño de la poblacion
numGeneraciones = 500 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)


#Algoritmo Genetico
tamPoblacion = 200 #Tamaño de la poblacion
numGeneraciones = 100 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)


#Algoritmo Genetico
tamPoblacion = 200 #Tamaño de la poblacion
numGeneraciones = 500 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)

#Algoritmo Genetico
tamPoblacion = 500 #Tamaño de la poblacion
numGeneraciones = 100 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)

#Algoritmo Genetico
tamPoblacion = 500 #Tamaño de la poblacion
numGeneraciones = 500 #Numero de generaciones (Condicion de terminacion)
maxReglas = 5 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)
Main.run(data_file,clasificador,about,estrategiaSimple,numParticiones,True,laplace,normalizar,separador)