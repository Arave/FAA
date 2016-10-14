# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori

print "Apartado 4 de la memoria- Fichero de datos crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 1"
estrategia=ValidacionCruzada(1)

clasificador=ClasificadorAPriori()
print "Prob. a priori para P(Class=+)"
prob1 = clasificador.probAPriori(dataset, "Class","+")
print prob1
print "Prob. a priori para P(Class=-)"
prob2 = clasificador.probAPriori(dataset, "Class","-")
print prob2