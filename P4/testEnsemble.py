# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorEnsemble
print "Practica 4 test de Clasificador Ensemble"

print "\nFichero de datos: ejemplo5.data"
dataset=Datos('./ConjuntosDatos/ejemplo5.data',True)
print "Estrategia: validacion cruzada, numParticiones: 5"
estrategia=ValidacionCruzada(5)
print "Clasificador: Ensamble"
clasificador = ClasificadorEnsemble()
print "Ejecucción: "
laplace = False
normalizar = True
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"