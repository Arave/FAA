# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorAPriori


print "\nApartado 4 de la memoria- Fichero de datos crx.data"
dataset=Datos('./ConjuntosDatos/crx.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
clasificador=ClasificadorAPriori()

errores=clasificador.validacionApartado(estrategia,dataset,clasificador,4)


print "\nApartado 4 de la memoria- Fichero de datos crx_clean.data"
dataset=Datos('./ConjuntosDatos/crx_clean.data',True)
print "Estrategia: validacion cruzada, numParticiones: 10"
estrategia=ValidacionCruzada(10)
clasificador=ClasificadorAPriori()

errores=clasificador.validacionApartado(estrategia,dataset,clasificador,4)
