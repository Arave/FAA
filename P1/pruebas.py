# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori


dataset=Datos('./ConjuntosDatosPruebas/d1.data',True)
estrategia=ValidacionCruzada(1)

#Probabilidades a priori: P(Class=positive) y P(Class=negative)
clasificador=ClasificadorAPriori()
print "Prob. a priori para P(Class=positive)"
prob1 = clasificador.probAPriori(dataset,"class","+")
print round(prob1,4)
print "Prob. a priori para P(Class=negative)"
prob2 = clasificador.probAPriori(dataset, "class", "-")
print round(prob2,4)


#Probabilidades de máxima verosimilitud: P(MLeftSq=b|Class=positive) P(TRightSq=x|Class=negative)
datos = dataset.datos
print "Prob. de máxima verosimilitud para P(x1=+|class=+)"
prob3 = clasificador.probMaxVerosimil(dataset, datos, "x1", "+", "class", "+")
print round(prob3,4)
print "Prob. de máxima verosimilitud para P(x2=-|class=-)"
prob4 = clasificador.probMaxVerosimil(dataset, datos,"x2", "-", "class", "-")
print round(prob4,4)