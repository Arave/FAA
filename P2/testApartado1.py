# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:51:02 2016

@author: Albert Soler, Alfonso Sebares
"""

from sklearn import preprocessing
#from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score

from Datos import Datos
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes
from Clasificador import ClasificadorAPriori

def genDataTxt(filename, data):
    fhandle = open(filename, 'w')
    fhandle.write("[")
    for row in data:
        fhandle.write("[")
        for item in row:
            fhandle.write("%.1f " % item)
        fhandle.write("]\n")
    fhandle.write("]")
    fhandle.close()
    return

def pruebasMatices():
    dataset = Datos('./ConjuntosDatos/tic-tac-toe.data', True)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
    x = encAtributos.fit_transform(dataset.datos[:, :-1])  # matriz de datos codificada
    # y = dataset.datos[:,-1]                                 #clase de cada patron

    # iris = datasets.load_iris()
    genDataTxt('datos_numpy.txt', dataset.datos)
    genDataTxt('datos_encoded.txt', x)
    # genDataTxt('datos_iris_data.txt', iris.data)
    return

def run(fichero_datos, cls, cls_brief, strat, k, supervisado, laplace):
    print "\nFichero de datos: " + fichero_datos
    dataset = Datos('./ConjuntosDatos/'+fichero_datos,supervisado)
    print "Estrategia: ",strat.getStratname(), ", numParticiones: ", str(k)
    print "Clasificador: ", cls_brief
    print "Ejecucion: "
    print "________________________________________________________________"
    errores = cls.validacion(strat, dataset, cls, laplace)
    print "\n"
    return errores

#NOTAS:
#   - Estructura enfocada a posibles modificaciones.
#   - Los que son full Nominales se pasan por HybridNB (OneHotEncoder+GaussianNB) como
#   indica el enunciado pero en principio se podría hacer en condiciones con MultinomialNB.
def runScikit(fichero_datos, cls, cls_brief, strat, cv, supervisado, laplace):
    dataset = Datos('./ConjuntosDatos/' + fichero_datos, supervisado)
    x = dataset.datos[:, :-1]
    y = dataset.datos[:, -1]
    clf = None
    if cls == "MultinomialNB":
        if laplace:
            alpha = 1.0
        else:
            alpha = 0
        clf = MultinomialNB(alpha, fit_prior=True,class_prior=None)
    elif cls == "GaussianNB":
        clf = GaussianNB()
    elif cls == "HybridNB":
        encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
        x = encAtributos.fit_transform(dataset.datos[:, :-1])  # matriz de datos codificada
        y = dataset.datos[:, -1]  # clase de cada patron
        # Doc: http://bit.ly/2eVXXcQ
        clf = GaussianNB()
    elif cls == "Prior":
        clf = DummyClassifier(strategy='prior')
    else:
        print "ERR: Clasificador no valido"
        return
    #siempre se aplica cross-validation con cv folds por defecto. Se puede cambiar si piden hold-out
    scores = cross_val_score(clf, x, y, cv=10)
    print "=================RESULTADO Scikit-learn===================="
    print("Overall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    """
    print "Media de errores total:", 100-scores.mean(), "%"
    print "Mediana de errores total:", 100-np.median(scores), "%"
    print "Desviación típica:", 100-scores.std(), "%"
    """
    return

print "Practica 2 Apartado 1 de la memoria"

pruebasMatices()

#Documentacion:
#http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#http://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators


#=======================================tic-tac-toe=======================================#
about = "clasificador a Priori"
run('tic-tac-toe.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
runScikit('tic-tac-toe.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
run('tic-tac-toe.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
runScikit('tic-tac-toe.data', "MultinomialNB", about, ValidacionCruzada(10),5,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
run('tic-tac-toe.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
runScikit('tic-tac-toe.data', "MultinomialNB", about, ValidacionCruzada(10),10,True,True)

#=======================================wine_proc=======================================#

about = "clasificador a Priori"
run('wine_proc.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
runScikit('wine_proc.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
run('wine_proc.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
runScikit('wine_proc.data', "GaussianNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
run('wine_proc.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
runScikit('wine_proc.data', "GaussianNB", about, ValidacionCruzada(10),10,True,True)

#==========================================crx==========================================#
about = "clasificador a Priori"
run('crx.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
runScikit('crx.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
run('crx.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
runScikit('crx.data', "HybridNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
run('crx.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
runScikit('crx.data', "HybridNB", about, ValidacionCruzada(10),10,True,True)

#=======================================crx_clean=======================================#
about = "clasificador a Priori"
run('crx_clean.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
runScikit('crx_clean.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
run('crx_clean.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
runScikit('crx_clean.data', "HybridNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
run('crx_clean.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
runScikit('crx_clean.data', "HybridNB", about, ValidacionCruzada(10),10,True,True)

#========================================digits========================================#
about = "clasificador a Priori"
run('digits.data', ClasificadorAPriori(), about, ValidacionCruzada(10),10,True,False)
runScikit('digits.data', "Prior", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes sin correción de Laplace"
run('digits.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,False)
runScikit('digits.data', "GaussianNB", about, ValidacionCruzada(10),10,True,False)

about = "clasificador Naive Bayes con correción de Laplace"
run('digits.data', ClasificadorNaiveBayes(), about, ValidacionCruzada(10),10,True,True)
runScikit('digits.data', "GaussianNB", about, ValidacionCruzada(10),10,True,True)
