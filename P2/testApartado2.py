# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Main import Main
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorVecinosProximos

print "Practica 2 test apartado 2"

#==================================================example3=================================================#
data_file = 'example3.data'
about = "clasificador a Vecinos próximos k=1"
clf_id = "KNeighborsClassifier"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 1)

about = "clasificador a Vecinos próximos k=3"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 3)

about = "clasificador a Vecinos próximos k=5"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 5)

about = "clasificador a Vecinos próximos k=11"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 11)

about = "clasificador a Vecinos próximos k=21"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 21)

about = "clasificador a Vecinos próximos k=51"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 51)

#normalizar=True
about = "clasificador a Vecinos próximos k=1, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 1)

about = "clasificador a Vecinos próximos k=3, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 3)

about = "clasificador a Vecinos próximos k=5, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 5)

about = "clasificador a Vecinos próximos k=11, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 11)

about = "clasificador a Vecinos próximos k=21, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 21)

about = "clasificador a Vecinos próximos k=51, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 51)

#==================================================example4=================================================#
data_file = 'example4.data'
about = "clasificador a Vecinos próximos k=1"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 1)

about = "clasificador a Vecinos próximos k=3"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 3)

about = "clasificador a Vecinos próximos k=5"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 5)

about = "clasificador a Vecinos próximos k=11"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 11)

about = "clasificador a Vecinos próximos k=21"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 21)

about = "clasificador a Vecinos próximos k=51"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 51)

#normalizar=True
about = "clasificador a Vecinos próximos k=1, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 1)

about = "clasificador a Vecinos próximos k=3, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 3)

about = "clasificador a Vecinos próximos k=5, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 5)

about = "clasificador a Vecinos próximos k=11, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 11)

about = "clasificador a Vecinos próximos k=21, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 21)

about = "clasificador a Vecinos próximos k=51, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 51)

#==================================================wine_proc=================================================#
data_file = 'wine_proc.data'
about = "clasificador a Vecinos próximos k=1"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 1)

about = "clasificador a Vecinos próximos k=3"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 3)

about = "clasificador a Vecinos próximos k=5"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 5)

about = "clasificador a Vecinos próximos k=11"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 11)

about = "clasificador a Vecinos próximos k=21"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 21)

about = "clasificador a Vecinos próximos k=51"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 51)

#normalizar=True
about = "clasificador a Vecinos próximos k=1, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 1)

about = "clasificador a Vecinos próximos k=3, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 3)

about = "clasificador a Vecinos próximos k=5, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 5)

about = "clasificador a Vecinos próximos k=11, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 11)

about = "clasificador a Vecinos próximos k=21, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 21)

about = "clasificador a Vecinos próximos k=51, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 51)


#==================================================wdbc=================================================#
data_file = 'wdbc.data'
about = "clasificador a Vecinos próximos k=1"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 1)

about = "clasificador a Vecinos próximos k=3"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 3)

about = "clasificador a Vecinos próximos k=5"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 5)

about = "clasificador a Vecinos próximos k=11"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 11)

about = "clasificador a Vecinos próximos k=21"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 21)

about = "clasificador a Vecinos próximos k=51"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 51)

#normalizar=True
about = "clasificador a Vecinos próximos k=1, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 1)

about = "clasificador a Vecinos próximos k=3, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 3)

about = "clasificador a Vecinos próximos k=5, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 5)

about = "clasificador a Vecinos próximos k=11, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 11)

about = "clasificador a Vecinos próximos k=21, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 21)

about = "clasificador a Vecinos próximos k=51, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 51)

#==================================================digits=================================================#
data_file='digits.data'
about = "clasificador a Vecinos próximos k=1"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 1)

about = "clasificador a Vecinos próximos k=3"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 3)

about = "clasificador a Vecinos próximos k=5"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 5)

about = "clasificador a Vecinos próximos k=11"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 11)

about = "clasificador a Vecinos próximos k=21"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 21)

about = "clasificador a Vecinos próximos k=51"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, False)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, False, 51)

#normalizar=True
about = "clasificador a Vecinos próximos k=1, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(1), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 1)

about = "clasificador a Vecinos próximos k=3, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(3), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 3)

about = "clasificador a Vecinos próximos k=5, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(5), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 5)

about = "clasificador a Vecinos próximos k=11, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(11), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 11)

about = "clasificador a Vecinos próximos k=21, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(21), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 21)

about = "clasificador a Vecinos próximos k=51, normalizar"
Main.run(data_file, ClasificadorVecinosProximos(51), about, ValidacionCruzada(10), 10, True, False, True)
Main.runScikit(data_file, clf_id, about, ValidacionCruzada(10),10,True,False, True, 51)