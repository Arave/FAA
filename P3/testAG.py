# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

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
###################################################################################
#default para pruebas:
mode = {'Prints':'verbose', 'Diversidad':'default', 'ReglasExtra':'randSons', 'Resto':'random'}

#test: maximiza el numero de reglas por individuo menos el ultimo:
#mode = {'Prints':'verbose', 'Diversidad':'maxReglas-1', 'Reglas_extra':'randSon'}


print "Practica 3 test AG"

#validacion Simple
numParticionesSimples = 1
porcentajeParticiones = 80
estrategia=ValidacionSimple(numParticionesSimples, porcentajeParticiones)

#Tweaks
laplace = False
normalizar = True
separador = False

#Algoritmo Genetico
tamPoblacion = 10 #Tamaño de la poblacion
numGeneraciones = 3 #Numero de generaciones (Condicion de terminacion)
maxReglas = 3 #Numero máximo de reglas por individuo
clasificador = AlgoritmoGenetico(tamPoblacion, numGeneraciones, maxReglas, mode)

#======================================pruebas_short_priori_B.data=====================================#
data_file = 'pruebas_short_priori_B.data'
about = "clasificador Algoritmo Genetico"
clf_id = "GenAlg"
Main.run(data_file,clasificador,about,estrategia,numParticionesSimples,True,laplace,normalizar,separador)
