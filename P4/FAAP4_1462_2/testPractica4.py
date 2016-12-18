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
from Clasificador import ClasificadorMulticlase


print "Practica 4 test de Clasificador Ensemble"

print "\nFichero de datos: wine_proc.data"
dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
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
clm = ClasificadorMulticlase(clf1)
clf2 = ClasificadorVecinosProximos(k)
clf3 = ClasificadorNaiveBayes()
arrayClf = [clm, clf2, clf3]

print "\nRESULTADOS CLASIFICADORES INDIVIDUALES - wine_proc.data: "
print "Clasificador: Naive Bayes"
clasificador = clf3
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Vecinos Proximos (7)"
clasificador = clf2
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Multiclase (Regresion Logistica)"
clasificador = clm
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Ensamble (Nuestra implementación)"
clasificador = ClasificadorEnsemble(arrayClf)
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


""" Segunda parte: VotingClassifier - Clasificadores Sklearn """ 
print "Clasificador: Ensamble Sklearn"
clasificador = ClasificadorEnsembleSklearn()
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "########################################################################################"


print "\nFichero de datos: tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 5"
estrategia=ValidacionCruzada(5)

laplace = False
normalizar = False

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

print "\nRESULTADOS CLASIFICADORES INDIVIDUALES - tic-tac-toe.data: "
print "Clasificador: Naive Bayes"
clasificador = clf3
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Vecinos Proximos (7)"
clasificador = clf2
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Regresion Logistica"
clasificador = clf1
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Ensamble (Nuestra implementación)"
clasificador = ClasificadorEnsemble(arrayClf)
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

""" Segunda parte: VotingClassifier - Clasificadores Sklearn """ 
print "Clasificador: Ensamble Sklearn"
clasificador = ClasificadorEnsembleSklearn()
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "###################################################################################"




print "\nFichero de datos: titanic.data"
dataset=Datos('./ConjuntosDatos/titanic.data',True)
print "Estrategia: validacion cruzada, numParticiones: 5"
estrategia=ValidacionCruzada(5)

laplace = False
normalizar = False

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

print "\nRESULTADOS CLASIFICADORES INDIVIDUALES - titanic.data: "
print "Clasificador: Naive Bayes"
clasificador = clf3
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Vecinos Proximos (7)"
clasificador = clf2
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Regresion Logistica"
clasificador = clf1
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Ensamble (Nuestra implementación)"
clasificador = ClasificadorEnsemble(arrayClf)
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


""" Segunda parte: VotingClassifier - Clasificadores Sklearn """ 
print "Clasificador: Ensamble Sklearn"
clasificador = ClasificadorEnsembleSklearn()
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "####################################################################################"





print "\nFichero de datos: digits.data"
dataset=Datos('./ConjuntosDatos/digits.data',True)
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
clm = ClasificadorMulticlase(clf1)
clf2 = ClasificadorVecinosProximos(k)
clf3 = ClasificadorNaiveBayes()
arrayClf = [clm, clf2, clf3]

print "\nRESULTADOS CLASIFICADORES INDIVIDUALES - digits.data: "
print "Clasificador: Naive Bayes"
clasificador = clf3
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Vecinos Proximos (7)"
clasificador = clf2
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"

print "Clasificador: Multiclase (Regresion Logistica)"
clasificador = clm
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
print "Clasificador: Ensamble (Nuestra implementación)"
clasificador = ClasificadorEnsemble(arrayClf)
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"


""" Segunda parte: VotingClassifier - Clasificadores Sklearn """ 
print "Clasificador: Ensamble Sklearn"
clasificador = ClasificadorEnsembleSklearn()
print "Ejecución: "
print "________________________________________________________________"
clasificador.validacion(estrategia,dataset,clasificador,laplace,normalizar)
print "\n"
#############################################################################################
