# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes
from Clasificador import ClasificadorAPriori

print "Apartado 1 de la memoria"

print "\nFichero de datos: tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
laplace = False
print "________________________________________________________________"
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes sin correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: tic-tac-toe.data"
dataset=Datos('./ConjuntosDatos/tic-tac-toe.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes con correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = True
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"



print "\nFichero de datos: wine_proc.data"
dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: wine_proc.data"
dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes sin correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: wine_proc.data"
dataset=Datos('./ConjuntosDatos/wine_proc.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes con correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = True
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"



print "\nFichero de datos: crx.data"
dataset=Datos('./ConjuntosDatos/crx.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: crx.data"
dataset=Datos('./ConjuntosDatos/crx.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes sin correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: crx.data"
dataset=Datos('./ConjuntosDatos/crx.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes con correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = True
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"



print "\nFichero de datos: crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes sin correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "\nFichero de datos: crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes con correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = True
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"



print "\nFichero de datos: digits.data"
dataset=Datos('./ConjuntosDatos/digits.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador a Priori"
clasificador = ClasificadorAPriori()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: digits.data"
dataset=Datos('./ConjuntosDatos/digits.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes sin correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = False
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"

print "Fichero de datos: digits.data"
dataset=Datos('./ConjuntosDatos/digits.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
print "Clasificador: clasificador Naive Bayes con correción de Laplace"
clasificador = ClasificadorNaiveBayes()
print "Ejecucción: "
print "________________________________________________________________"
laplace = True
errores=clasificador.validacion(estrategia,dataset,clasificador,laplace)
print "\n"
