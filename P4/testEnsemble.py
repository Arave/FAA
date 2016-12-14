# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorEnsemble
from Clasificador import ClasificadorEnsembleSklearn
from Clasificador import ClasificadorRegresionLogistica
from Clasificador import ClasificadorVecinosProximos
from Clasificador import ClasificadorNaiveBayes
print "Practica 4 test de Clasificador Ensemble"

print "\nFichero de datos: ejemplo5.data"
dataset=Datos('./ConjuntosDatos/pruebas_short_priori.data',True)
print "Estrategia: validacion cruzada, numParticiones: 5"
estrategia=ValidacionCruzada(5)

laplace = False
normalizar = True




""" Primera parte: Hacerlo con nuestros clasificadores """

#Variables RegLog
nEpocas = 10
cteAprendizaje = 1
#Variables Knn
k = 7

clf1 = ClasificadorRegresionLogistica(nEpocas, cteAprendizaje) 
clf2 = ClasificadorVecinosProximos(k)
clf3 = ClasificadorNaiveBayes()
arrayClf = [clf1, clf2, clf3]

print "Clasificador: Ensamble (Nuestra implementación)"
clasificador = ClasificadorEnsemble(arrayClf)
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"





""" Segunda parte: VotingClassifier - Clasificadores Sklearn """ 
"""
print "Clasificador: Ensamble Sklearn"
clasificador = ClasificadorEnsembleSklearn()
print "Ejecucción: "
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
"""