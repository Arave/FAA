# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import ClasificadorAPriori

print "Prueba 1- Fichero de datos d1.data"
dataset=Datos('./ConjuntosDatosPruebas/d1.data',True)
print dataset.datos
"""
print "Estrategia: validación simple, numParticionesSimples 10, 80% train"
estrategia=ValidacionSimple(10,80)
#print estrategia.nombreEstrategia
print "Clasificador: clasificador a priori"
clasificador=ClasificadorAPriori()
print "Errores: "
#NOTA:
errores=clasificador.validacion(estrategia,dataset,clasificador)
"""