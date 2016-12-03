from __future__ import division #Para divisiones float por defecto
from collections import Counter
import numpy as np
import operator

from Clasificador import Clasificador

class AlgoritmoGenetico(Clasificador):
    probCruce = 60  # Probabilidad de cruzar individuos 60%
    probMutacionBit = 0.1  # prob. mutación de 1 bit 0.1%
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

    def __init__(self, tamPoblacion, numGeneraciones, maxReglas, debug):
        self.tamPoblacion = tamPoblacion
        self.numGeneraciones = numGeneraciones
        self.maxReglas = maxReglas
        self.debug = debug

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

    """Funcion que selecciona individuos (numSeleccionar) en base a su fitness
        también llamada ruleta creo, método visto en teoría. Se realiza un array de
        tam 100 y cada individo ocupa en relación a su fitness, cuanto más fitness más ocupa
        Y al crear un num. aleatorio entre 0 y numSeleccionar, es más probable que caiga en
        los que mayor fitness tienen.
        Ejemplo: A: f=10, B:f=5, C:f=3, D:f=2 ==> AAAAAAAAAABBBBBCCCDD más o menos
        Como implementarlo da igual, lo importante es que sea proporcional al fitness"""

    def seleccionProporcionalFitness(self, poblacion, fitness, numSeleccionar, sizeRegla):

        # print "========= DEBUG: seleccionProporcionalFitness ========"
        # Array a devover
        seleccionados = np.zeros(shape=(numSeleccionar, self.maxReglas, sizeRegla))

        # Calcular el fitness total de todos los individuos
        fitnessTotal = float(sum(fitness))
        # si el fitnessTotal es 0, cada individuo tiene fitness 0, devolver  ind. aleatorios
        if fitnessTotal == 0.0:
            print "Fitness de cada individuo = 0.0, devolviendo numSeleccionar indiv. aleatorios de la poblacion anterior"
            seleccionados = poblacion[np.random.choice(numSeleccionar, numSeleccionar, replace=False)]
            return seleccionados

            # Calcular el fitness relativo de cada individuo dado fitnessTotal
        fitnessRelativo = [f / fitnessTotal for f in fitness]
        # Generar los intervalos de probabilidad para cada individuo
        probs = [sum(fitnessRelativo[:i + 1]) for i in range(len(fitnessRelativo))]

        # print "fitness poblacion", fitness,"\nFitness Total (suma):", fitnessTotal
        # print "fitness relativo", fitnessRelativo,"\n Probs:", probs

        # Seleccionar numSeleccionar individuos
        if self.debug:
            print "in \"Seleccionar numSeleccionar individuos\":"
        for n in xrange(numSeleccionar):  # n - índice de individuos seleccionados
            r = np.random.rand()
            # Reccorer poblacion
            for i in xrange(self.tamPoblacion):  # i - índice de la poblacion de indiv. a seleccionar
                # print "% r Random:",r, "Prob. seleccionar indi.(",i,")",probs[i]
                if r <= probs[i]:  # elegimos el primer ind. cuyo porcentaje sea mayor al aleatorio que hemos generado
                    seleccionados[n] = poblacion[i]
                    if self.debug:
                        print "\t[i]", i
                        print "\tfitness[i]", fitness[i]
                    break
        return seleccionados

    """Funcion que cruza en un punto padre y madre y devuelve 2 hijos"""

    def cruceEnUnPunto(self, padre, madre, tipo="lossy"):
        if self.debug:
            print "in \"cruceEnUnPunto()\":"
            print "Padre:\n", padre
            print "Madre:\n", madre
        numReglasPadre = np.count_nonzero(~np.isnan(padre)) // self.sizeRegla
        numReglasMadre = np.count_nonzero(~np.isnan(madre)) // self.sizeRegla
        numReglas = 0
        if numReglasMadre > numReglasPadre:
            numReglas = numReglasPadre
        else:
            numReglas = numReglasMadre

        hijo1 = np.zeros(shape=(self.maxReglas, self.sizeRegla))
        hijo2 = np.zeros(shape=(self.maxReglas, self.sizeRegla))

        for i in xrange(numReglas):
            index1 = np.random.randint(1, self.sizeRegla - 2)
            if self.debug:
                print "index1 de cruce: ", index1
            hijo1[i] = np.concatenate((padre[i][:index1], madre[i][index1:]))
            hijo2[i] = np.concatenate((madre[i][:index1], padre[i][index1:]))
            np.squeeze(hijo1[i])
            np.squeeze(hijo2[i])

        """¿TODO?: añadir diferencia de reglas si numReglasP != numReglasM,
        las copiamos aleat a los hijos?"""

        # Poner a None las reglas vacías
        while numReglas < self.maxReglas:
            hijo1[numReglas] = None
            hijo2[numReglas] = None
            numReglas += 1

        if self.debug:
            print "hijo1:\n", hijo1
            print "hijo2:\n", hijo2
        return hijo1, hijo2

    """ Falta editarlo, cuando funcione bien cruceEnUnPunto, se termina este
    def cruceEnDosPuntos(self, padre, madre):

        index1 = np.random.randint(1, self.sizeRegla - 2)
        index2 = np.random.randint(1, self.sizeRegla - 2)
        if index1 > index2:
            index1, index2 = index2, index1
        hijo1 = padre[:index1] + madre[index1:index2] + padre[index2:]
        hijo2 = madre[:index1] + padre[index1:index2] + madre[index2:]
        return (hijo1, hijo2) """

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        # 1º- Inicializar una población aleatoria de individuos

        # Tamaño de cada regla es fijo = Suma del núm. valores posibles por atributos + 1 (clase)
        # sizeRegla = Counter(chain.from_iterable(e.keys() for e in diccionario))
        sizeRegla = 0
        for d in diccionario:
            sizeRegla += len(d)
        sizeRegla = sizeRegla - 1  # Restar uno de la clase. la clase (bin) se mapea como 0 o 1. no como 2 bits
        self.sizeRegla = sizeRegla
        poblacion = self.inicializarPoblacion(self.tamPoblacion, sizeRegla)
        if self.debug:
            print "\nPoblacion 0:\n", poblacion

        # Evaluar el fitness de la población inicial
        fitness = self.calcularFitness(poblacion, datostrain, atributosDiscretos, diccionario)
        if self.debug:
            print "Valor de fitness de la poblacion inicial", fitness, "\n"

        newPoblacion = np.zeros(shape=(self.tamPoblacion, self.maxReglas, sizeRegla))

        # mientras no se satisfazca la condicion de terminacion
        for i in xrange(self.numGeneraciones):

            contadorNewPoblacion = 0  # Contador de elementos insertados en newPoblacion

            # Pasar a los mejores a la sigueiten poblacion - Elitismo. El porcentaje que nos digan
            numElitistas = (self.propElitismo * self.tamPoblacion) // 100
            if numElitistas == 0:
                numElitistas = 1  # Si el porcentaje es muy pequeño, pasar al menos 1.
            indicesE = np.argpartition(fitness, -numElitistas)[-numElitistas:]
            for idx in xrange(numElitistas):  # idx - índice de elististas
                newPoblacion[idx] = poblacion[indicesE[idx]]  # copiarlos
            if self.debug:
                print "new poblacion (post elitismo) [Array todo nan -> individuo empty]:\n", newPoblacion, "\n"

            contadorNewPoblacion += numElitistas

            # Seleccion de individuos respecto a una condicion. En nuestro caso,
            # proporcional fitness
            numCruce = (self.probCruce * self.tamPoblacion) // 100
            if numCruce % 2 != 0:  # Si el numero de elementos a cruzar es impar, convertirlo a par
                numCruce += 1
            if self.tipoSeleccion == "Proporcional al fitness":
                seleccionados = self.seleccionProporcionalFitness(poblacion, fitness, numCruce, sizeRegla)
                if self.debug:
                    print "\nSeleccionados (cruce):\n", seleccionados, "\n"
                indiceCruce = 0
                while indiceCruce < numCruce:
                    # print "contadorNewPoblacion: ", contadorNewPoblacion
                    hijo1, hijo2 = self.cruceEnUnPunto(seleccionados[indiceCruce], seleccionados[indiceCruce + 1])
                    # print "hijo1 \n",hijo1
                    # print "hijo2 \n",hijo2
                    newPoblacion[contadorNewPoblacion], newPoblacion[contadorNewPoblacion + 1] = hijo1, hijo2
                    contadorNewPoblacion += 2
                    indiceCruce += 2
                if self.debug:
                    print "\nnew poblacion (after cruce) [Array todo 0, muy posiblmente individuo empty]:\n", newPoblacion, "\n"

            # Mutacion
            numMutacion = 0

            fitness = self.calcularFitness(newPoblacion, datostrain, atributosDiscretos, diccionario)
            indexMayorFitness, value = max(enumerate(fitness), key=operator.itemgetter(1))
            print "Fitness Mejor individuo: ", fitness[indexMayorFitness]
            print "Regla(s) Mejor individuo: ", poblacion[indexMayorFitness]

            """TODO:si se da la prob. de mutar, mutar uno de los individuos resultantes de los seleccionados ? o de 10 random?.

            [si poblacion = 100 individuos ] Ahora habría 60 (cruce) + 5 (elitismo) en la newPoblacion. =>
            Opcion1. Evaluar el fitness de estos (65 nuevos) y añadirle 35 (100 - 65) individuos de la anterior población hasta llegar a tamPoblacion.
            Opcion2. Añadir (100 - 65) individuos de la anteriorhasta llegar a tamPoblacion, y ahora calcular el fitness de los individuos. Problema, estás
            recalculando el fitness de 35 indidividuos que ya habías calculado antes.

            OJO: El tamPoblacion siempre debe ser el mismo
            OJO2: Cuanto mayor Fitness Mejor es el individuo

            Continuar con el bucle hasta condición de parada, en nuestro caso numGeneraciones. """

            poblacion = newPoblacion
            fitness = self.calcularFitness(poblacion, datostrain, atributosDiscretos, diccionario)
            print "Valor de fitness de la poblacion final Generacion (", i, ")", fitness, "\n"
            if self.debug:
                print "\n===================================== gen (", i, ") ends =====================================\n"

        print "------------  Fin de entrenamiento -----------\n\n"
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