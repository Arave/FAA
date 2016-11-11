# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Main import Main
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorRegresionLogistica

print "Practica 2 test apartado 3"
nEpocas = 1
cteAprendizaje = 1
clasificador = ClasificadorRegresionLogistica(nEpocas, cteAprendizaje)

#Fichero 1 - example3.data
#==================================================example3=================================================#
data_file = 'example3.data'
about = "clasificador Regresion Logistica"
clf_id = "LogisticRegression"

Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False)

#normalizar=True
about = "clasificador Regresion Logistica, normalizar"
Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True)

#==================================================example4=================================================#
data_file = 'example4.data'
about = "clasificador Regresion Logistica"

Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False)

#normalizar=True
about = "clasificador Regresion Logistica, normalizar"
Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True)

#==================================================wdbc=================================================#
data_file = 'wdbc.data'

Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False)

#normalizar=True
about = "clasificador Regresion Logistica, normalizar"
Main.run(data_file, clasificador, about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True)

#==================================================wine=================================================#


#=================================================digits================================================#

