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

#Probabilidades a priori: P(Class=positive) y P(Class=negative)
print "Prob. a priori para P(Class=positive)"
prob1 = clasificador.probAPriori(dataset,"Class","positive")
print round(prob1,4)
print "Prob. a priori para P(Class=negative)"
prob2 = clasificador.probAPriori(dataset, "Class", "negative")
print round(prob2,4)


#Probabilidades de máxima verosimilitud: P(MLeftSq=b|Class=positive) P(TRightSq=x|Class=negative)
datos = dataset.datos
print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
prob3 = clasificador.probMaxVerosimil(dataset, datos,"MLeftSq", "b", "Class", "positive")
print round(prob3,4)
print "Prob. de máxima verosimilitud para P(TRightSq=x|Class=negative)"
prob4 = clasificador.probMaxVerosimil(dataset, datos,"TRightSq", "x", "Class", "negative")
print round(prob4,4)