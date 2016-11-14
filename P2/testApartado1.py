# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from Main import Main
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes
from Clasificador import ClasificadorAPriori


print "Practica 2 Apartado 1 de la memoria"

#=======================================tic-tac-toe=======================================#
about = "clasificador a Priori"
Main.run('tic-tac-toe.data', ClasificadorAPriori(), about, ValidacionCruzada(10), 10, True, False)
Main.runScikit('tic-tac-toe.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
Main.run('tic-tac-toe.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('tic-tac-toe.data', "MultinomialNB", about, ValidacionCruzada(10),5,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
Main.run('tic-tac-toe.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
Main.runScikit('tic-tac-toe.data', "MultinomialNB", about, ValidacionCruzada(10),10,True,True)

#=======================================wine_proc=======================================#
about = "clasificador a Priori"
Main.run('wine_proc.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('wine_proc.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
Main.run('wine_proc.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('wine_proc.data', "GaussianNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
Main.run('wine_proc.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
Main.runScikit('wine_proc.data', "GaussianNB", about, ValidacionCruzada(10),10,True,True)

#==========================================crx==========================================#
about = "clasificador a Priori"
Main.run('crx.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('crx.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
Main.run('crx.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('crx.data', "HybridNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
Main.run('crx.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
Main.runScikit('crx.data', "HybridNB", about, ValidacionCruzada(10),10,True,True)

#=======================================crx_clean=======================================#
about = "clasificador a Priori"
Main.run('crx_clean.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('crx_clean.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
Main.run('crx_clean.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('crx_clean.data', "HybridNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
Main.run('crx_clean.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
Main.runScikit('crx_clean.data', "HybridNB", about, ValidacionCruzada(10),10,True,True)

#========================================digits========================================#
about = "clasificador a Priori"
Main.run('digits.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('digits.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
Main.run('digits.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
Main.runScikit('digits.data', "GaussianNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
Main.run('digits.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
Main.runScikit('digits.data', "GaussianNB", about, ValidacionCruzada(10),10,True,True)