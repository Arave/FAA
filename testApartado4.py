# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori
"""Para el conjunto credit approval mostrar los siguientes valores únicamente 
para la primera partición (validación cruzada):"""
print "Apartado 4 de la memoria- Fichero de datos crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 1"
estrategia=ValidacionCruzada(1)
clasificador=ClasificadorAPriori()

#Probabilidades a priori: P(Class=+) y P(Class=-)
print "Prob. a priori para P(Class=+)"
prob = clasificador.probAPriori(dataset, "Class","+")
print round(prob,4)
print "Prob. a priori para P(Class=-)"
prob = clasificador.probAPriori(dataset, "Class","-")
print round(prob,4)

datos = dataset.datos
#Probabilidades de máxima verosimilitud: P(A7=bb|Class=+) P(A4=u|Class=-)
print "Prob. de máxima verosimilitud para P(A7=bb|Class=+)"
prob = clasificador.probMaxVerosimil(dataset, datos, "A7", "bb","Class","+")
print round(prob,4)
print "Prob. de máxima verosimilitud para P(A4=u|Class=-)"
prob = clasificador.probMaxVerosimil(dataset, datos, "A4", "u","Class","-")
print round(prob,4)



#Media y desviación típica de los atributos continuos A2, A14 y A15 
#condicionado al valor de clase +
print "Media y desviación típica  A2 condicionado a clase +"
media, std = clasificador.mediaDesviacionAtr(dataset, "A2","Class","+")
print media
print std

print "Media y desviación típica  A14 condicionado a clase +"

media, std = clasificador.mediaDesviacionAtr(dataset, "A14","Class","+")
print media
print std
print "Media y desviación típica  A15 condicionado a clase +"

media, std = clasificador.mediaDesviacionAtr(dataset, "A15","Class","+")
print media
print std
