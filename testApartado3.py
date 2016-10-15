# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori

print "Apartado 3 de la memoria- Fichero de datos tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 1"
estrategia=ValidacionCruzada(1)

clasificador=ClasificadorAPriori()
print "Prob. a priori para P(Class=positive)"
prob1 = clasificador.probAPriori(dataset,"Class","positive")
print prob1
print "Prob. a priori para P(Class=negative)"
prob2 = clasificador.probAPriori(dataset, "Class", "negative")
print prob2