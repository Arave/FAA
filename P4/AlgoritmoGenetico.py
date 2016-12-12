# -*- coding: utf-8 -*-
from __future__ import division #Para divisiones float por defecto
from collections import Counter
import copy
import numpy as np
import operator

from Clasificador import Clasificador
from plotter import genPlot

class AlgoritmoGenetico(Clasificador):
    probCruce = 60  # Probabilidad de cruzar individuos 60%
    probMutacionBit = 0.1  #ESTÁTICO, prob. mutación de 1 bit 0.1%
    probMutacion = 10  # Probabilidad de mutar 10 %
    propElitismo = 5  # Proporcion de individuos que pasan gracias al elitismo 5%
    tipoSeleccion = "Proporcional al fitness"  # Se realiza una seleccion proporcional al fitness
    sizeRegla = 0

    # ARGS!
    tamPoblacion = 10  # Tamaño de la poblacion
    numGeneraciones = 100  # Numero de generaciones (Condicion de terminacion)
    maxReglas = 10  # Numero máximo de reglas por individuo

    # Es lo que utiliza clasifica
    bestIndividuo = None  # Mejor individuo -> Se utilizará para clasificar

    def __init__(self, tamPoblacion, numGeneraciones, maxReglas, mode):
        self.tamPoblacion = tamPoblacion
        self.numGeneraciones = numGeneraciones
        self.maxReglas = maxReglas
        self.mode = mode

    """Funcion que permite inicizalizar una poblacion aleatoria de individuos"""

    def inicializarPoblacion(self, tamPoblacion, sizeRegla):
        # Poblacion array en 3D. tamaño población, número máximo reglas, y tamaño de regla
        # Aquellos individuos que no tengan todas ls reglas posibles, su regla en vez de
        # ser un array de 0-1, será un 0.
        poblacion = np.zeros(shape=(tamPoblacion, self.maxReglas, sizeRegla))

        # Recorrer la poblacion
        for idx in xrange(tamPoblacion):
            # Determinar el numero de reglas por individuo = numReglas
            numReglas = np.random.randint(low=1, high=self.maxReglas + 1, size=1)

            if self.mode['Diversidad']=='maxReglas-1':
                numReglas = self.maxReglas
                if idx == tamPoblacion-1:
                    numReglas = 1

            # print "numReglas", numReglas
            # Rellener las reglas del individuo de manera aleatoria
            for i in xrange(numReglas):
                # Generar reglas random
                poblacion[idx][i] = np.random.randint(2, size=sizeRegla)

            # Poner a None las reglas vacías
            while numReglas < self.maxReglas:
                poblacion[idx][numReglas] = None
                numReglas += 1
                # Contar el número de reglas que tiene el array
                # Contar no NaN: https://stackoverflow.com/questions/21778118/counting-the-number-of-non-nan-elements-in-a-numpy-ndarray-matrix-in-python
                # print np.count_nonzero(~np.isnan(poblacion[idx])) // sizeRegla
        return poblacion

    @staticmethod
    def valorFitness(datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula fitness del individuo
        # print "====DEBUG: valorFitness===="
        numColumnas = datos.shape[1]
        numFilas = datos.shape[0]
        numAciertos = 0
        arrayEqual = np.equal(datos[:, numColumnas - 1], pred)
        # print "datos:",datos[:,numColumnas-1],"\n pred",pred

        numAciertos = np.sum(arrayEqual)  # Contar los True
        # print "numAciertos", numAciertos, "numFilas: ", numFilas
        return (numAciertos / numFilas) * 100

    """Funcion que permite obtener el fitness de la poblacion"""

    def calcularFitness(self, poblacion, datostrain, atributosDiscretos, diccionario):
        ret = np.zeros(shape=self.tamPoblacion)
        for idx in xrange(self.tamPoblacion):
            self.bestIndividuo = poblacion[idx]  # "Sugerencia: llamada a clasifica con cada individuo de la población"
            predicciones = self.clasifica(datostrain, atributosDiscretos, diccionario)
            fitnessVal = self.valorFitness(datostrain, predicciones)
            ret[idx] = fitnessVal
            # print "Valor de fitness", fitnessVal
        return ret

    #Selecciona un numero N de individuos de manera aleatoria
    def seleccionAleatoria(self, poblacion, numSeleccionar, sizeRegla):
        # (!) seleccionados = np.zeros(shape=(numSeleccionar, self.maxReglas, sizeRegla))
        seleccionados = poblacion[np.random.choice(numSeleccionar, numSeleccionar, replace=False)]
        return seleccionados

        #Funcion que selecciona individuos (numSeleccionar) en base a su fitness
        #también llamada ruleta creo, método visto en teoría. Se realiza un array de
        #tam 100 y cada individo ocupa en relación a su fitness, cuanto más fitness más ocupa
        #Y al crear un num. aleatorio entre 0 y numSeleccionar, es más probable que caiga en
        #los que mayor fitness tienen.
        #Ejemplo: A: f=10, B:f=5, C:f=3, D:f=2 ==> AAAAAAAAAABBBBBCCCDD más o menos
        #Como implementarlo da igual, lo importante es que sea proporcional al fitness
    def seleccionProporcionalFitness(self, poblacion, fitness, numSeleccionar, sizeRegla):
        # Array a devover
        seleccionados = np.zeros(shape=(numSeleccionar, self.maxReglas, sizeRegla))

        # Calcular el fitness total de todos los individuos
        fitnessTotal = float(sum(fitness))
        # si el fitnessTotal es 0, cada individuo tiene fitness 0, devolver  ind. aleatorios
        if fitnessTotal == 0.0:
            if self.mode['Prints'] == 'verbose':
                print "\tFitness de cada individuo = 0.0, devolviendo numSeleccionar indiv. aleatorios de la poblacion anterior"
            seleccionados = poblacion[np.random.choice(numSeleccionar, numSeleccionar, replace=False)]
            return seleccionados

            # Calcular el fitness relativo de cada individuo dado fitnessTotal
        fitnessRelativo = [f / fitnessTotal for f in fitness]
        # Generar los intervalos de probabilidad para cada individuo
        probs = [sum(fitnessRelativo[:i + 1]) for i in range(len(fitnessRelativo))]

        # print "fitness poblacion", fitness,"\nFitness Total (suma):", fitnessTotal
        # print "fitness relativo", fitnessRelativo,"\n Probs:", probs

        # Seleccionar numSeleccionar individuos
        if self.mode['Prints'] == "verbose":
            print "in \"Seleccionar numSeleccionar individuos\":"
        for n in xrange(numSeleccionar):  # n - índice de individuos seleccionados
            r = np.random.rand()
            # Reccorer poblacion
            for i in xrange(self.tamPoblacion):  # i - índice de la poblacion de indiv. a seleccionar
                # print "% r Random:",r, "Prob. seleccionar indi.(",i,")",probs[i]
                if r <= probs[i]:  # elegimos el primer ind. cuyo porcentaje sea mayor al aleatorio que hemos generado
                    seleccionados[n] = poblacion[i]
                    if self.mode['Prints'] == "verbose":
                        print "\t[i]", i
                        print "\tfitness[i]", fitness[i]
                    break
        return seleccionados

    #Funcion que cruza en un punto padre y madre y devuelve 2 hijos
    def cruceEnUnPunto(self, padre, madre):
        if self.mode['Prints'] == "verbose":
            print "in \"cruceEnUnPunto()\":"
            print "Padre:\n", padre
            print "Madre:\n", madre

        numReglasPadre = np.count_nonzero(~np.isnan(padre)) // self.sizeRegla
        numReglasMadre = np.count_nonzero(~np.isnan(madre)) // self.sizeRegla
        numReglas = 0
        diff = 0
        flag_equal = None
        if numReglasMadre > numReglasPadre:
            numReglas = numReglasPadre
            diff = numReglasMadre - numReglasPadre
        elif numReglasMadre < numReglasPadre:
            numReglas = numReglasMadre
            diff = numReglasPadre - numReglasMadre
        else:
            numReglas = numReglasMadre
            diff = 0
            flag_equal = "iguales"

        hijo1 = np.zeros(shape=(self.maxReglas, self.sizeRegla))
        hijo2 = np.zeros(shape=(self.maxReglas, self.sizeRegla))

        for i in xrange(numReglas):
            index1 = np.random.randint(1, self.sizeRegla - 2)
            if self.mode['Prints'] == "verbose":
                print "\tindex1 de cruce(rand): ", index1
            hijo1[i] = np.concatenate((padre[i][:index1], madre[i][index1:]))
            hijo2[i] = np.concatenate((madre[i][:index1], padre[i][index1:]))
            np.squeeze(hijo1[i])
            np.squeeze(hijo2[i])

        #repartir de forma aleatoria entre los hijos als reglas del padre/madre sobrantes
        numReglasH1 = numReglas
        numReglasH2 = numReglas
        if flag_equal is None:
            if self.mode['ReglasExtra'] == "randSons":
                #caso: repartir reglas extra del padre
                if numReglasPadre > numReglasMadre:
                    for d in xrange(diff):
                        hijo = np.random.randint(1, 3)
                        if hijo == 1:
                            hijo1[numReglas+d] = padre[numReglas+d]
                            numReglasH1 += 1
                        else:
                            hijo2[numReglas+d] = padre[numReglas+d]
                            numReglasH2 += 1
                #caso: repartir reglas extra de la madre
                else:
                    for d in xrange(diff):
                        hijo = np.random.randint(1, 3)
                        if hijo == 1:
                            hijo1[numReglas+d] = madre[numReglas+d]
                            numReglasH1 += 1
                        else:
                            hijo2[numReglas+d] = madre[numReglas+d]
                            numReglasH2 += 1
            #  self.mode['Reglas_extra'] == 'default':
            else:
                #default, se trunca el numero de reglas de los hijos
                pass

        # Poner a None las reglas vacías
        while numReglasH1 < self.maxReglas:
            hijo1[numReglasH1] = None
            numReglasH1 +=1
        while numReglasH2 < self.maxReglas:
            hijo2[numReglasH2] = None
            numReglasH2 +=1

        if self.mode['Prints'] == "verbose":
            print "hijo1:\n", hijo1
            print "hijo2:\n", hijo2
        return hijo1, hijo2

    """
    def cruceEnDosPuntos(self, padre, madre):
        index1 = np.random.randint(1, self.sizeRegla - 2)
        index2 = np.random.randint(1, self.sizeRegla - 2)
        if index1 > index2:
            index1, index2 = index2, index1
        hijo1 = padre[:index1] + madre[index1:index2] + padre[index2:]
        hijo2 = madre[:index1] + padre[index1:index2] + madre[index2:]
        return hijo1, hijo2
    """
    #def seleccionProporcionalFitness(self, poblacion, fitness, numSeleccionar, sizeRegla):
    #Muta con las probabilidades definidas en el objeto.
    #La probabilidad de mutar un bit es fija al 0,1%
    def mutar(self, poblacion, numMutaciones, sizeRegla):
        seleccionados = np.zeros(shape=(numMutaciones, self.maxReglas, sizeRegla))
        seleccionados_aleat = self.seleccionAleatoria(poblacion, numMutaciones, sizeRegla)
        if self.mode['Prints'] == 'verbose':
            print "Seleccionado/s aleat. (",numMutaciones, ") para posible mutacion:\n",seleccionados_aleat
        for idx, indv in enumerate(seleccionados_aleat):
            seleccionados[idx] = indv
            muta = np.random.randint(1, 101) #1%
            muta = 1
            if muta == 1: #equiprobabilidad en los 100000 nums
                if self.mode['Prints'] == 'verbose':
                    print "\t[MUTACION TUVO LUGAR!]"
                    print "\tIndividuo previo mutacion:", seleccionados[idx]
                numReglas = np.count_nonzero(~np.isnan(indv)) // sizeRegla
                muta_regla = np.random.randint(0, numReglas)
                muta_bit = np.random.randint(0,sizeRegla) #rand del bit a mutar
                if seleccionados[idx][muta_regla][muta_bit] == 0:
                    seleccionados[idx][muta_regla][muta_bit] = 1
                else:
                    seleccionados[idx][muta_regla][muta_bit] = 0
                if self.mode['Prints']=='verbose':
                    print "\tIndividuo post mutacion:",seleccionados[idx]
        return seleccionados

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario, plot_flag=None):
        # 1º- Inicializar una población aleatoria de individuos

        # Tamaño de cada regla es fijo = Suma del núm. valores posibles por atributos + 1 (clase)
        # sizeRegla = Counter(chain.from_iterable(e.keys() for e in diccionario))
        sizeRegla = 0
        #plot
        bestIndvGens = []
        fitnessMedioGens = []
        print "Diccionarios atributos:", diccionario[:-1]," clase: ", diccionario[-1],"\n"
        for d in diccionario:
            sizeRegla += len(d)
        sizeRegla -= 1  # Restar uno de la clase. la clase (bin) se mapea como 0 o 1. no como 2 bits
        self.sizeRegla = sizeRegla
        poblacion = self.inicializarPoblacion(self.tamPoblacion, sizeRegla)
        if self.mode['Prints'] == "verbose":
            print "\nPoblacion 0:\n", poblacion

        # Evaluar el fitness de la población inicial
        # 1ª EVAL DEL FITNESS
        fitness = self.calcularFitness(poblacion, datostrain, atributosDiscretos, diccionario)
        if self.mode['Prints'] == "verbose":
            print "Valor de fitness de la poblacion inicial", fitness, "\n"

        indexMayorFitness, value = max(enumerate(fitness), key=operator.itemgetter(1))
        mejorFitness = fitness[indexMayorFitness]
        repesCount = 1
        # mientras no se satisfazca la condicion de terminacion
        for i in xrange(self.numGeneraciones):
            newPoblacion = np.zeros(shape=(self.tamPoblacion, self.maxReglas, sizeRegla))
            contadorNewPoblacion = 0  # Contador de elementos insertados en newPoblacion
            # Pasar a los mejores a la sigueiten poblacion - Elitismo. El porcentaje que nos digan
            numElitistas = (self.propElitismo * self.tamPoblacion) // 100

            if numElitistas == 0:
                numElitistas = 1  # Si el porcentaje es muy pequeño, pasar al menos 1.
            indicesE = np.argpartition(fitness, -numElitistas)[-numElitistas:]

            if self.mode['Prints'] == 'verbose':
                print "\n====>COMIENZO GEN (",i,")"
                print "numElitistas: ", numElitistas, "indicesE: ", indicesE
                print "[contadorNewPoblacion: ",contadorNewPoblacion,"]\n"

            for idx in xrange(numElitistas):  # idx - índice de elististas
                newPoblacion[idx] = poblacion[indicesE[idx]]  # copiarlos
            if self.mode['Prints'] == 'verbose':
                print "new poblacion (post elitismo) [Array todo nan -> individuo empty]:\n", newPoblacion, "\n"

            contadorNewPoblacion += numElitistas
            if self.mode['Prints']=='verbose':
                print "[contadorNewPoblacion: ",contadorNewPoblacion,"]\n"
            # Seleccion de individuos respecto a una condicion. En nuestro caso,
            # proporcional fitness
            numCruce = (self.probCruce * self.tamPoblacion) // 100
            if numCruce % 2 != 0:  # Si el numero de elementos a cruzar es impar, convertirlo a par
                numCruce += 1
            if self.tipoSeleccion == "Proporcional al fitness":
                seleccionados = self.seleccionProporcionalFitness(poblacion, fitness, numCruce, sizeRegla)
                if self.mode['Prints'] == "verbose":
                    print "\nSeleccionados cruce (", numCruce, "):\n", seleccionados, "\n"
                indiceCruce = 0
                while indiceCruce < numCruce:
                    # print "contadorNewPoblacion: ", contadorNewPoblacion
                    hijo1, hijo2 = self.cruceEnUnPunto(seleccionados[indiceCruce], seleccionados[indiceCruce + 1])
                    # print "hijo1 \n",hijo1
                    # print "hijo2 \n",hijo2
                    newPoblacion[contadorNewPoblacion], newPoblacion[contadorNewPoblacion + 1] = hijo1, hijo2
                    contadorNewPoblacion += 2
                    indiceCruce += 2
                if self.mode['Prints'] == "verbose":
                    print "\nnew poblacion (after cruce) [Array todo 0, muy posiblmente individuo empty]:\n", newPoblacion, "\n"
                    print "[contadorNewPoblacion: ", contadorNewPoblacion, "]\n"

            # Mutacion
            #comprobacion por si se toma un tam de poblacion muy pequeño
            #Ej.: tam 5 - con elitismo y cruce ya llega a 5 por el ajuste de 0 y ajuste de impares
            if contadorNewPoblacion < self.tamPoblacion:
                numMutaciones = (self.probMutacion * self.tamPoblacion) // 100
                if numMutaciones == 0:
                    numMutaciones = 1  # Si el porcentaje es muy pequeño, pasar al menos 1.
                seleccionados = self.mutar(poblacion,numMutaciones, self.sizeRegla)
                for idx, idv in enumerate(seleccionados):
                    newPoblacion[contadorNewPoblacion] = seleccionados[idx]
                    contadorNewPoblacion += 1
                if self.mode['Prints'] == "verbose":
                    print "[contadorNewPoblacion (post mutacion): ", contadorNewPoblacion, "]\n"

            #Resto: distintos criterios para rellenar los ultimos que falten
            #Misma comprobacion para tamPoblacion pequeños
            if contadorNewPoblacion < self.tamPoblacion:
                restantes = self.tamPoblacion-contadorNewPoblacion
                if self.mode['Resto'] == 'random':
                    seleccionados = self.seleccionAleatoria(poblacion,restantes,self.sizeRegla)
                    if self.mode['Prints'] == 'verbose':
                        print "\nSeleccionados aleatorios para fill (",restantes, "):\n", seleccionados
                elif self.mode['Resto'] == 'fitness':
                    seleccionados = self.seleccionProporcionalFitness(poblacion, fitness, restantes, self.sizeRegla)
                    if self.mode['Prints'] == 'verbose':
                        print "\nSeleccionados prop. al fitness para fill (",restantes, "):\n", seleccionados
                else:
                    #posible implementacion de otro criterio
                    pass
                for idx,chosen in enumerate(seleccionados):
                    newPoblacion[contadorNewPoblacion] = seleccionados[idx]
                    contadorNewPoblacion += 1
                if self.mode['Prints'] == 'verbose':
                    print "[contadorNewPoblacion (post fill): ", contadorNewPoblacion, "]\n"

            #2ª EVAL DEL FITNESS (necesaria por las nuevas inserciones. Podría optimizarse por índices)
            fitness = self.calcularFitness(newPoblacion, datostrain, atributosDiscretos, diccionario)
            indexMayorFitness, value = max(enumerate(fitness), key=operator.itemgetter(1))
            print "Fitness Mejor individuo: ", fitness[indexMayorFitness]
            print "Regla(s) Mejor individuo: \n \t", poblacion[indexMayorFitness]

            #plot
            bestIndvGens.append(fitness[indexMayorFitness])
            fitnessMedioGens.append(np.median(fitness))

            poblacion = copy.deepcopy(newPoblacion)

            #Ya calculado arriba
            #fitness = self.calcularFitness(poblacion, datostrain, atributosDiscretos, diccionario)
            print "Valor de fitness de la poblacion al final Generacion (", i, ")", fitness, "\n"
            if self.mode['Prints'] == "verbose":
                print "Poblacion al final de la gen(",i,"):\n", poblacion
                print "\n===================================== gen (", i, ") ends =====================================\n"
            #Si el fitness del mejor individuo es 100 (no hay errores de clasificación hemos terminado)
            if self.mode['CondicionTerminacion'] != "no":
                if fitness[indexMayorFitness] >= int(self.mode['CondicionTerminacion']):
                    print "El fitness del mejor individuo es mayor a la condición de terminación. Finalizando entrenamiento.."
                    break

            if self.mode['Threshold'] is not None and fitness[indexMayorFitness] >= self.mode['Threshold']:
                if mejorFitness == fitness[indexMayorFitness]:
                    repesCount += 1
                else:
                    mejorFitness = fitness[indexMayorFitness]
                    repesCount = 1
                if repesCount == 10:
                    if self.mode['Prints'] == 'verbose':
                        print "[!] Mejor fitness repetido 10 veces: fin de entrenamiento"
                    break
        #(fin loop generaciones)
        print "------------  Fin de entrenamiento -----------\n\n"
        #plot
        if plot_flag == True:
            genPlot(None,bestIndvGens,fitnessMedioGens)

        # Ya hemos "entrenado" los individuos, ahora simplemente cogemos el mejor individo
        indexMayorFitness, value = max(enumerate(fitness), key=operator.itemgetter(1))
        self.bestIndividuo = poblacion[indexMayorFitness]

    # Clase 0 por defecto cuando no hay match
    def clasifica(self, datostest, atributosDiscretos, diccionario, correcion=None):
        # Evaluar reglas del individuo
        numFilas = datostest.shape[0]
        numColumnas = datostest.shape[1]
        ret = np.zeros(shape=numFilas)
        resultadoDefecto = 0.0  # Resultado por defecto

        # print "============ DEBUG: Clasifica ==============="
        # print "Individuo: ", self.bestIndividuo

        # Recorrer todos los datos Test (instancias)
        for idx in xrange(numFilas):  # idx - índice de cada instancia (fila) del test
            # print "--Instancia de Test(",idx,")",datostest[idx]
            # Recorrer todas las reglas del mejor individuo
            prediReglas = []  # Array de clases que predice 1 individuo por cada instancia del test(como mucho una clase por regla)
            numReglas = np.count_nonzero(~np.isnan(self.bestIndividuo)) // self.sizeRegla
            for i in xrange(numReglas):
                # Evaluar regla
                flagCoincide = 1  # Coincide la regla
                # Reccorrer todos los atributos de la instancia del test, para comprobar si están en la regla.
                numBitsSaltar = 0
                # print "--REGLA (",i,")",self.bestIndividuo[i]
                for atr in xrange(numColumnas - 1):  # que NO mire en el atr. de la clase
                    valorAtributo = int(datostest[idx][atr])
                    # Comprobar si NO hay un uno para ese valor --> sigueine regla
                    if self.bestIndividuo[i][numBitsSaltar + valorAtributo] != 1.0:
                        flagCoincide = 0
                        # print "No coincide en la regla",i, "atributo: ", atr
                    # Numero de bits a saltar (anteriores atributos). Máximo posible de representación en bits de los pasados atr.
                    numBitsSaltar += len(diccionario[atr])
                    # print "numBitsSaltar: ", numBitsSaltar, "atributo: ", atr
                if flagCoincide == 1:  # AND implicita
                    predClaseIndi = self.bestIndividuo[i][-1]  # Coge el último bit, predice la clase
                    prediReglas.append(predClaseIndi)
                    # print "Coincide en todas, predice clase: ", predClaseIndi
            # print "Array de clases predecidas: ", prediReglas
            # Si ninguna regla ha predicho nada, asignar clase por defecto
            if len(prediReglas) == 0:
                ret[idx] = resultadoDefecto
                # print "ninguna regla ha predicho nada, asignado clase default",resultadoDefecto
            else:
                most_common, num_most_common = Counter(prediReglas).most_common(1)[0]
                ret[idx] = most_common
                # print "Array de clases predecidas: ", prediReglas, "clase mayoritaria: ", most_common

        # print "Predicciones:", ret
        # print "============ END debug clasifica ==============="

        return ret  # devolver el array de predicciones