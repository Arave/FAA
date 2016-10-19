# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori

print "Prueba 2- Fichero de datos d2.data"
dataset=Datos('./ConjuntosDatosPruebas/d2.data',True)
print "Estrategia: validacion cruzada, numParticiones: 5"
estrategia=ValidacionCruzada(5)
#print estrategia.nombreEstrategia
print "Clasificador: clasificador a priori"
clasificador=ClasificadorAPriori()
print "Errores: "
#NOTA:
errores=clasificador.validacion(estrategia,dataset,clasificador)
