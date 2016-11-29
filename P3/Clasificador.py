# -*- coding: utf-8 -*-
from __future__ import division #Para divisiones float por defecto
from operator import itemgetter
from collections import Counter
from scipy.special import expit
from plotModel import plotModel
from copy import copy,deepcopy
from itertools import chain
import copy
import numpy as np
import math
import operator

#import scipy.stats


class Clasificador(object):
  # Clase abstracta
  #__metaclass__ = ABCMeta
  plotCount = 0

  def clasifica(self,datosTest,atributosDiscretos,diccionario, correcion=None):
      scores = self.score(datosTest,atributosDiscretos,diccionario, correcion)
      return np.argmax(scores,axis=1)

  # devuelve una matriz numpy con el score para cada clase y dato
  def score(self,datosTest,atributosDiscretos,diccionario, correcion=None): 
      scores = np.zeros((len(datosTest),len(diccionario[-1])))
      preds = map(lambda x: int(x), self.clasifica(datosTest,atributosDiscretos,diccionario, correcion))
      scores[range(datosTest.shape[0]),preds] = 1.0
      return scores

  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  @staticmethod
  def error(datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error   
      numColumnas = datos.shape[1]
      numFilas = datos.shape[0]
      numAciertos = 0
      numFallos = 0
      arrayEqual = np.equal(datos[:,numColumnas-1],pred)
      numAciertos = np.sum(arrayEqual) #Contar los True
      numFallos =  numFilas - numAciertos
      return (numFallos / numFilas) * 100
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  @staticmethod
  def validacion(particionado,dataset,clasificador,correcionL=False,normalizacion=False,seed=None, plotName=None):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
      
    
       particiones = particionado.creaParticiones(dataset.datos)
       arrayErrores = np.empty(particionado.numeroParticiones)

       """
       if particionado.nombreEstrategia == "ValidacionSimple":
           print "Indices train y test para [" + str(particionado.numeroParticiones) + "] particiones:"
       elif particionado.nombreEstrategia == "ValidacionCruzada":
           print 'Datos de train y test para [', particionado.numeroParticiones,'] grupos:'
       else:
           print "ERR: nombre de estrategia no valido"
           exit(1)
       print 'Correción de Laplace:',correcionL
       print 'Normalizar:',normalizacion
       """
       #for each particion: clasificar y sacar los errores de cada evaluación
       for idx, p in enumerate(particiones):
           #print "======================================================"
           #print "PARTICION (" + str(idx) + "):"
           #print "======================================================"
           #print p
           datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])

           #Normalizar las datos si flag normalizacion = True
           if normalizacion == True:
               #Obtener media y std para cada atributo
               #print 'Datos train sin normalizar' , datosTrain
               #print '\nbefore:\n', datosTrain
               dataset.calcularMediasDesv(datosTrain)
               dataset.normalizarDatos(datosTrain)
               #print '\nafter:\n', datosTrain
               #print 'Datos train normalizados' , datosTrain
               #print 'Datos test sin normalizar' ,datosTest 
               dataset.calcularMediasDesv(datosTest)
               dataset.normalizarDatos(datosTest)
               #print 'Datos test normalizados' ,datosTest 

           print ' =>DatosTrain [', idx, ']:'
           print datosTrain
           print ' =>DatosTest [', idx, ']:'
           print datosTest

           # Entrenamiento
           clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
           pred = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios,correcionL)
           #print "Predicción: "
           #print pred

           error = clasificador.error(datosTest, pred)
           arrayErrores[idx] = error
          
           #print "\t Porcentaje de error (%): ",error
           #estrategia=ValidacionSimple(10,80) => particionado, arg[0] - numero de particiones. Calcular la media y desv.
       #estadística
       print "=================RESULTADO===================="  
       print "Array de % de errores obtenidos:" ,arrayErrores, ""
       print "Media de errores total:" ,np.mean(arrayErrores), "%"
       print "Mediana de errores total:" ,np.median(arrayErrores), "%"
       print "Desviación típica:" ,np.std(arrayErrores), "%"

       if isinstance(clasificador, ClasificadorVecinosProximos) or isinstance(clasificador, ClasificadorRegresionLogistica):
           ii = particiones[-1].indicesTrain
           # clasificador = ClasificadorVecinosProximos(1)
           # print plotName
           plotModel(dataset.datos[ii, 0], dataset.datos[ii, 1], dataset.datos[ii, -1] != 0, clasificador, "Frontera", plotName)
#################################################################################################################

class AlgoritmoGenetico(Clasificador):
    probCruce = 60 #Probabilidad de cruzar individuos 60%
    probMutacion = 0.1 #Probabilidad de mutación de un individuo 0.1%
    propElitismo = 5 #Proporcion de individuos que pasan gracias al elitismo 5%
    tipoSeleccion = "Proporcional al fitness" #Se realiza una seleccion proporcional al fitness

    #ARGS!
    tamPoblacion = 10 #Tamaño de la poblacion
    numGeneraciones = 100 #Numero de generaciones (Condicion de terminacion)
    maxReglas = 10 #Numero máximo de reglas por individuo
    
    bestIndividuo = None #Mejor individuo -> Se utilizará para clasificar


    def __init__(self, tamPoblacion, numGeneraciones, maxReglas):
        self.tamPoblacion = tamPoblacion
        self.numGeneraciones = numGeneraciones
        self.maxReglas = maxReglas

    """Funcion que permite inicizalizar una poblacion aleatoria de individuos"""
    def inicializarPoblacion(self, tamPoblacion, sizeRegla):
        
        #Poblacion array en 3D. tamaño población, número máximo reglas, y tamaño de regla
        #Aquellos individuos que no tengan todas ls reglas posibles, su regla en vez de 
        #ser un array de 0-1, será un 0. 
        poblacion = np.zeros(shape=(tamPoblacion, self.maxReglas, sizeRegla))
        
        #Recorrer la poblacion
        for idx in xrange(tamPoblacion):
            #Determinar el numero de reglas por individuo = numReglas
            numReglas = np.random.randint(low=1, high=self.maxReglas + 1, size=1)
            #Rellener las reglas del individuo de manera aleatoria            
            for i in xrange(numReglas):
                #Generar reglas random
                poblacion[idx][i] = np.random.randint(2, size=sizeRegla)  
        return poblacion        

    @staticmethod
    def valorFitness(datos,pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula fitness del individuo 
        print "====DEBUG: valorFitness===="
        numColumnas = datos.shape[1]
        numFilas = datos.shape[0]
        numAciertos = 0
        arrayEqual = np.equal(datos[:,numColumnas-1],pred)
        print "datos:",datos[:,numColumnas-1],"\n pred",pred

        numAciertos = np.sum(arrayEqual) #Contar los True
        print "numAciertos", numAciertos, "numFilas: ", numFilas
        return (numAciertos / numFilas) * 100        
            
        
    """Funcion que permite obtener el fitness de la poblacion"""
    def calcularFitness(self, poblacion, datostrain, atributosDiscretos, diccionario):
        ret = np.zeros(shape=self.tamPoblacion)
        for idx in xrange(self.tamPoblacion):       
            self.bestIndividuo = poblacion[idx] #"Sugerencia: llamada a clasifica con cada individuo de la población"
            predicciones = self.clasifica(datostrain, atributosDiscretos, diccionario)
            fitnessVal = self.valorFitness(datostrain, predicciones)
            ret[idx] = fitnessVal
            #print "Valor de fitness", fitnessVal
        return ret

    """Funcion que selecciona individuos (numSeleccionar) en base a su fitness
        también llamada ruleta creo, método visto en teoría. Se realiza un array de
        tam 100 y cada individo ocupa en relación a su fitness, cuanto más fitness más ocupa
        Y al crear un num. aleatorio entre 0 y numSeleccionar, es más probable que caiga en
        los que mayor fitness tienen.
        Ejemplo: A: f=10, B:f=5, C:f=3, D:f=2 ==> AAAAAAAAAABBBBBCCCDD más o menos
        Como implementarlo da igual, lo importante es que sea proporcional al fitness"""        
    def seleccionProporcionalFitness(self, poblacion, fitness, numSeleccionar, sizeRegla):
        fitnessTotal = float(sum(fitness))
        print "fitnessTotal: ", fitnessTotal, "\n"
        fitnessRelativo = [f/fitnessTotal for f in fitness]
        # Generar los intervalos de probabilidad para cada individuo
        probs = [sum(fitnessRelativo[:i+1]) for i in range(len(fitnessRelativo))]
        # Seleccionar numSeleccionar individuos
        seleccionados = np.zeros(shape=(numSeleccionar, self.maxReglas, sizeRegla))
        for n in xrange(numSeleccionar):
            r = np.random.randint(low=0, high=1, size=1)
            for i in xrange(self.tamPoblacion):     
                if r <= probs[i]:
                    seleccionados[n] = poblacion[i]
                    break
        return seleccionados
        
    
    """Funcion que cruza en un punto padre y madre y devuelve 2 hijos"""
    @staticmethod        
    def cruceEnUnPunto(padre, madre):
        pass    
        
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        #1º- Inicializar una población aleatoria de individuos
    
        # Tamaño de cada regla es fijo = Suma del núm. valores posibles por atributos + 1 (clase)
        #sizeRegla = Counter(chain.from_iterable(e.keys() for e in diccionario))
        sizeRegla = 0            
        for d in diccionario:
            sizeRegla += len(d)
        sizeRegla = sizeRegla - 1 #Restar uno de la clase. la clase (bin) se mapea como 0 o 1. no como 2 bits    
        poblacion = self.inicializarPoblacion(self.tamPoblacion,sizeRegla)
        print "Poblacion 0:\n", poblacion
        
        #Evaluar el fitness de la población inicial
        fitness = self.calcularFitness(poblacion, datostrain, atributosDiscretos, diccionario)
        print "Valor de fitness de la poblacion 0", fitness, "\n"
        
        newPoblacion = np.zeros(shape=(self.tamPoblacion, self.maxReglas, sizeRegla))
        #mientras no se satisfazca la condicion de terminacion
        for i in xrange(self.numGeneraciones):
            #Pasar a los mejores a la sigueiten poblacion - Elitismo. El porcentaje que nos digan
            numElitistas = (self.propElitismo * self.tamPoblacion) // 100
            if numElitistas == 0:
                numElitistas = 1 #Si el porcentaje es muy pequeño, pasar al menos 1. 
            indicesE = np.argpartition(fitness, -numElitistas)[-numElitistas:]
            for idx in xrange(numElitistas):
                newPoblacion[idx] = poblacion[indicesE[idx]] #copiarlos
                fitness[indicesE[idx]] = 0.0 #poner a 0 el fitness, "eliminarlos"
            print "new poblacion (after fitness):\n", newPoblacion, "\n"

            #Seleccion de individuos respecto a una condicion. En nuestro caso,
            #proporcional fitness
            numCruce = (self.probCruce * self.tamPoblacion) // 100
            if self.tipoSeleccion == "Proporcional al fitness":
                seleccionados = self.seleccionProporcionalFitness(poblacion, fitness, numCruce, sizeRegla)

                print "Seleccionados (cruce)\n:", seleccionados, "\n"
                """TODO: Seleccionar el self.probCruce % de población que los individuos seleccionados por: seleccionProporcionalFitness
                y cruzarlos, y los hijos asignarlos a newPoblacion.
                Y si se da la prob. de mutar, mutar uno de los individuos resultantes de los seleccionados. 
                
                [si poblacion = 100 individuos ] Ahora habría 60 (cruce) + 5 (elitismo) en la newPoblacion. =>
                Opcion1. Evaluar el fitness de estos (65 nuevos) y añadirle 35 (100 - 65) individuos de la anterior población hasta llegar a tamPoblacion.
                Opcion2. Añadir (100 - 65) individuos de la anteriorhasta llegar a tamPoblacion, y ahora calcular el fitness de los individuos. Problema, estás
                recalculando el fitness de 35 indidividuos que ya habías calculado antes. 
                
                OJO: El tamPoblacion siempre debe ser el mismo
                OJO2: Cuanto mayor Fitness Mejor es el individuo
                
                Continuar con el bucle hasta condición de parada, en nuestro caso numGeneraciones. """
                
            pass
        
        """TODO: Ya hemos "entrenado" los individuos, ahora simplemente cogemos el mejor individo y lo copiamos:
            self.bestIndividuo = poblacion[indexMayorFitness] y hemos terminado"""

            

        
    #Clase 0 por defecto cuando no hay match
    def clasifica(self, datostest, atributosDiscretos, diccionario, correcion=None):
        #Evaluar reglas del individuo
        numFilas = datostest.shape[0]
        numColumnas = datostest.shape[1]
        ret = np.zeros(shape=numFilas)
        resultadoDefecto = 0.0 #Resultado por defecto
        
        print "============ DEBUG: Clasifica ==============="
        print "Individuo: ", self.bestIndividuo

        #Recorrer todos los datos Test (instancias)
        for idx in xrange(numFilas): #idx - índice de cada instancia (fila) del test
            print "--Instancia de Test(",idx,")",datostest[idx]
            #Recorrer todas las reglas del mejor individuo
            prediReglas = [] #Array de clases que predice 1 individuo por cada instancia del test(como mucho una clase por regla)
            for i in xrange(self.maxReglas): #i - índice de reglas
                #Evaluar regla
                flagCoincide = 1 #Coincide la regla
                #Reccorrer todos los atributos de la instancia del test, para comprobar si están en la regla.
                numBitsSaltar = 0
                print "--REGLA (",i,")",self.bestIndividuo[i]
                for atr in xrange(numColumnas - 1): #que NO mire en el atr. de la clase
                    valorAtributo = int(datostest[idx][atr])
                    #Comprobar si NO hay un uno para ese valor --> sigueine regla
                    if self.bestIndividuo[i][numBitsSaltar + valorAtributo] != 1.0:
                        flagCoincide = 0
                        print "No coincide en la regla",i, "atributo: ", atr
                    #Numero de bits a saltar (anteriores atributos). Máximo posible de representación en bits de los pasados atr.
                    numBitsSaltar += len(diccionario[atr])
                    #print "numBitsSaltar: ", numBitsSaltar, "atributo: ", atr    
                if flagCoincide == 1: #AND implicita
                    predClaseIndi = self.bestIndividuo[i][-1] #Coge el último bit, predice la clase
                    prediReglas.append(predClaseIndi)
                    print "Coincide en todas, predice clase: ", predClaseIndi
            print "Array de clases predecidas: ", prediReglas        
            #Si ninguna regla ha predicho nada, asignar clase por defecto
            if len(prediReglas) == 0:
                ret[idx] = resultadoDefecto
                print "ninguna regla ha predicho nada, asignado clase default",resultadoDefecto
            else:    
                most_common,num_most_common = Counter(prediReglas).most_common(1)[0]    
                ret[idx] = most_common
                print "Array de clases predecidas: ", prediReglas, "clase mayoritaria: ", most_common   
        
        print "Predicciones:", ret
        print "============ END debug clasifica ==============="
            
        return ret #devolver el array de predicciones


                            
                
        
        
    


class ClasificadorRegresionLogistica(Clasificador):
    datostrain = None
    nEpocas = 0
    cteAprendizaje = 0
    vectorW = 0

    def __init__(self, nEpocas, cteAprendizaje):
        self.nEpocas = nEpocas
        self.cteAprendizaje = cteAprendizaje
        
    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):
        numFilas = datostrain.shape[0]
        numColumnas = datostrain.shape[1]

        #generar un vector W, con valores aleatorios entre -0.5 y 0.5 para
        #numColumnas - 1 + 1 (X0)
        vectorW = np.random.uniform(low=-0.5, high=0.5, size=(numColumnas,))
        #para ie=1 -> nEpocas
        for ie in xrange(self.nEpocas):
            #para i=1 -> N
            for i in xrange(numFilas):
                #Calcular Sigmoidal i (Sig(vectorWt * vectorXi)) === P(C1|vectorXi)  - Posteriori
                #Primero calculamos la vectorWt * vectorXi
                instanceTrain = np.insert(datostrain[i], 0, 1) #Añadir 1 al inicio
                #Multiplicar los elementos del array
                sumatorio = 0
                sumatorio = np.dot(vectorW, instanceTrain[:-1])
                """for m in xrange(numColumnas):
                    sumatorio += vectorW[m] * instanceTrain[m]"""
                """print 'vectorW', vectorW
                print 'instanceTrain', instanceTrain[:-1]
                print 'mult', sumatorio"""
                #Calcular la sigmoidal
                sig = expit(sumatorio)
                #Calcular el nuevo vectorW
                #vectorW = vectorW - cteAprendizaje(sig - Ti) * (vectorXi)
                vectorW = vectorW - (self.cteAprendizaje*(sig - (1 - instanceTrain[-1])))* instanceTrain[:-1]
        #self.vectorW = np.linalg.norm(vectorW)
        self.vectorW = vectorW 

        
    def score(self,datosTest,atributosDiscretos,diccionario, correcion=None): 
        numFilas = datosTest.shape[0]
        numColumnas = datosTest.shape[1]
        #predicciones = []
        ret = np.zeros(shape=(numFilas,2))

        for i in xrange(numFilas):
            #Calcular Sigmoidal i (Sig(vectorWt * vectorXi)) === P(C1|vectorXi)  - Posteriori
            #Primero calculamos la vectorWt * vectorXi
            instanceTrain = np.insert(datosTest[i], 0, 1) #Añadir 1 al inicio
            #Multiplicar los elementos del array
            sumatorio = 0        
            #sumatorio = np.dot(self.vectorW, instanceTrain[:-1])
            for m in xrange(numColumnas):
                sumatorio += self.vectorW[m] * instanceTrain[m]
            #Calcular la sigmoidal
            sig = expit(sumatorio)
            ret[i][0] = sig  
            ret[i][1] = 1 - sig 
            """#Comprobar a que clase pertenece
            if sig > 0.5:
                predicciones.append(0)
            else:
                predicciones.append(1)"""
        return ret    
            

##############################################################################


class ClasificadorVecinosProximos(Clasificador):
    datostrain = None
    k = 0 #numero de vecinos

    def __init__(self, k):
        self.k = k

    # Calcula la distancia ecluidea entre 2 instancias de tam length. Es decir, 
    #la raiz cuadrada de la suma de las diferencias(resta) entre 2 arrays de 
    #números (Instance 1 y 2).
    @staticmethod
    def distanciaEuclidea(instance1, instance2, length):
        distance = 0
        for x in xrange(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    # Función que devuelve los K vecinos más cercanos/similares para una fila de Test
    # llamada testInstance respecto a datosTrain. 
    @staticmethod
    def getVecinos(datostrain, testInstance, k):
        distancias = []
        length = len(testInstance) - 1 #La última es la clase
        #Recorrer todos las filas de entrenamiento para allar las distancias
        for x in xrange(len(datostrain)):
            #Calcular la distancia ecluidea del Test respecto a la la fila X del conjunto train. 
            dist = ClasificadorVecinosProximos.distanciaEuclidea(testInstance, datostrain[x], length)
            distancias.append((datostrain[x], dist))
        distancias.sort(key=operator.itemgetter(1)) #Ordenar las distancias de menos a mayor
        vecinos = [] #Vecinos más cercanos
        #Hasta el num. de vecinos, ir añadiendo las menores distancias
        for x in xrange(k):
            vecinos.append(distancias[x][0])
        return vecinos

    # Obtiene la clase mayoría si cada Vecino vota la suya (dice cual es más
    # probable)
    @staticmethod
    def getResultado(vecinos):
        classVotes = {} #Diccionario con las clases y los votos 
        #Reccorrer para todos los vecinos y obtener la clase que vota cada uno
        for x in xrange(len(vecinos)):
            response = vecinos[x][-1]
            if response in classVotes: #Si ya existe la clase sumarle 1 al "contador"
                classVotes[response] += 1
            else: #No existe esa clase, inicializarla a 1, ese voto. 
                classVotes[response] = 1
        #Ordenar los votos de mayor a menor (reverse=True)
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0] #Devolver la clase mayoritaria

    def entrenamiento(self, datostrain, atributosDiscretos=None, diccionario=None):
        #Guardar los datos de train para usarlos en test al calcular distancias
        self.datostrain = datostrain  
        

    def clasifica(self, datostest, atributosDiscretos=None, diccionario=None, correcion=None):
        numFilas = datostest.shape[0]
        ret = np.zeros(shape=(numFilas))

        # Recorrer los datos(filas) del test
        for x in xrange(numFilas):
            #Obtener los vecinos para la fila X delos datos del test
            vecinos = self.getVecinos(self.datostrain, datostest[x], self.k)
            #Obtener la clase mayoritaria para esa fila
            resultado = self.getResultado(vecinos)
            #Añadirla al array de predicciones
            #predicciones.append(resultado)
            ret[x] = resultado
        return ret #devolver el array de predicciones


#############################################################################

class ClasificadorAPriori(Clasificador):
  
  mayoritaria=0

  def entrenamiento(self,datostrain,atributosDiscretos=None,diccionario=None):
    # Obtener la clase mayoritaria de los datos
      numColumnas = datostrain.shape[1]
      most_common,num_most_common = Counter(datostrain[:,numColumnas-1]).most_common(1)[0]        
      #print most_common
      #print num_most_common
      
      self.mayoritaria = most_common
      return most_common     
    
  def clasifica(self,datostest,atributosDiscretos=None,diccionario=None,correcion=None):
    # Asignar la clase mayoritaria a todos los datos
      numFilas = datostest.shape[0]
      datos = np.empty(numFilas)
      datos.fill(self.mayoritaria)
      return datos

  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  tablaValores = []
  tablaMedia = []
  tablaStd = []  
  arrayPriori = []

  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      
     #print "============= Entrenamiento Naive Bayes ========================"
     numColumnas = datostrain.shape[1]
     idxColumnaClase = numColumnas - 1
     clases = diccionario[idxColumnaClase]
     sorted_x = sorted(clases.items(), key=operator.itemgetter(1))
     tabla = (len(atributosDiscretos) - 1)*[None] #No guardamos una tabla de clase
     tablaM = (len(atributosDiscretos) - 1)*[None]
     tablaS = (len(atributosDiscretos) - 1)*[None]
     arrayP = []
     arrayM = []
     arrayS = []
     
     del self.tablaValores[:] #Limpiar la lista de ejecucciones anteriores
     del self.arrayPriori[:] #Limpiar la lista de ejecucciones anteriores
     del self.tablaMedia[:] #Limpiar la lista de ejecucciones anteriores
     del self.tablaStd[:] #Limpiar la lista de ejecucciones anteriores
     
     #Recorrer las clases
     #for key, value in sorted(mydict.iteritems(), key=lambda (k,v): (v,k)):
     for key, value in sorted_x:      
         #Calcular los a priori
         probP = self.probAPriori(datostrain, diccionario, idxColumnaClase, key)
         arrayP.append(probP) 
   
     #Recorrer los atributos excepto el último (Clase)
     for idx,atr in enumerate(atributosDiscretos[:-1]):
         if atr == True: #nominal/discreto            
             #contar núm de ocurrencias para cada valor del atributo en cada clase         
             arrayC = []
             for key, value in sorted_x:
                 #contar atributos => rellenar tabla
                 cont = self.contarAtributos(datostrain, diccionario, idx, idxColumnaClase, key)
                 arrayC.append(cont)               
             tabla[idx] = arrayC
         else: #continuo
             for key, value in sorted_x:
                 #Calcular la media y std para las clases
                 media, std = self.mediaDesviacionAtr(datostrain, diccionario, idx, idxColumnaClase, key)
                 arrayM.append(media)
                 arrayS.append(std)
             tablaM[idx] = arrayM
             tablaS[idx] = arrayS
     
     self.tablaValores = tabla
     self.arrayPriori = arrayP
     self.tablaMedia = tablaM
     self.tablaStd = tablaS
     #print "sorted_x", sorted_x
     #print "tablaM", tablaM
     #print "t  ablaS", tablaS
     #print 'tablaValores train plana:\n',self.tablaValores
     #print 'prob. priori de clases:\n', self.arrayPriori
     """
     print "Tabla de valores (None=atributo continuo):"
     for t in self.tablaValores:
         print "\t",t, ""
     print "Array a priori:" ,self.arrayPriori, ""    
     print "Array media:" ,self.arrayMedia, "" 
     print "Array desviación típica (STD):" ,self.arrayStd, ""
     """

  #devuelve una copia de la tabla con la correción de Laplace aplicada
  @staticmethod
  def corregirTabla(tablaValores):
      t_copy = copy.deepcopy(tablaValores)
      for i_f,fila in enumerate(t_copy):
          if fila is not None:
              for i_c,clase in enumerate(fila):
                  for i_v,value in enumerate(clase):
                      t_copy[i_f][i_c][i_v] += 1
      return t_copy

  #devuelve una copia de la tabla normalizada
  @staticmethod
  def normalizarTabla(tablaValores):
      t_copy = copy.deepcopy(tablaValores)
      for i_f,fila in enumerate(t_copy):
          if fila is not None:
              for i_c,clase in enumerate(fila):
                  cst_clase = clase[:]
                  for i_v,value in enumerate(clase):
                      #print 'clase:', cst_clase
                      #print 'value:',value
                      #print 'value/sum(clase):',value/sum(cst_clase)
                      t_copy[i_f][i_c][i_v] = value/sum(cst_clase)
                      #print 't_copy[i_f][i_c][i_v]=',t_copy[i_f][i_c][i_v]
      return t_copy

  #PDF: densidad de probabilidad de una distribucion normal/gaussiana
  #(version 'math' en caso de que haya problema de compatibilidad)
  #uso: f(x|media,varianza)
  @staticmethod
  def normpdf(x, mean, sd):
      var = float(sd) ** 2
      pi = 3.1415926
      denom = (2 * pi * var) ** .5
      num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
      return num / denom

#http://naivebayes.blogspot.com.es/2013/05/clasificador-naive-bayes-como-funciona.html
  def clasifica(self,datostest,atributosDiscretos,diccionario, correcionL=False):
      posteriori = []
      numColumnas = datostest.shape[1]
      idxColumnaClase = numColumnas - 1
      #el orden de 'clases' coincide con el orden de 'self.arrayPriori'
      clases = diccionario[idxColumnaClase]
      sorted_clases = sorted(clases.items(), key=operator.itemgetter(1))
      #print "================= CLASIFICA ===================================="
      #print 'clases en clasifica',clases
      #print 'datostest:\n',datostest
      #print 'tablaValores:\n',self.tablaValores

      #print 'Correción de Laplace:',correcionL
      if correcionL: #en funcion de lo que pases desde validacion(), aplica correcion o no
          self.tablaValores = self.corregirTabla(self.tablaValores)
          #print 'tablaValores train (corregida):\n', self.tablaValores
      self.tablaValores = self.normalizarTabla(self.tablaValores)

      #print 'tablaValores train (norm,corregida):\n',self.tablaValores
      for tupla in datostest:
          pred = self.evalua(tupla,sorted_clases,atributosDiscretos)
          posteriori.append(pred)

      return posteriori

  #evalua una tupla de datosTest y devuelve la clase con mas probabilidad
  def evalua(self, tupla, sorted_clases, atributosDiscretos):
      ##print "===========EVALUA==================="
      ##print '\ttupla:',tupla
      #bucle 1: recorrer por clase
      arg = []
      #print 'tablaValores train (norm,corregida):\n', self.tablaValores
      for clase,idx_clase in sorted_clases:   #key=clase, value=idx_clase 
          sumatorio = 0.0
          probClase = 0.0 
          probClase = self.arrayPriori[idx_clase]
          for idx_atri, atri in enumerate(self.tablaValores):
              prob = 0.0
              if atributosDiscretos[idx_atri]: #caso discreto
                  #sacar valor del atributo de test y pasarlo a indice (int)
                  value = int(round(tupla[idx_atri]))
                  #print 'value:',value
                  #hacer el match con la tabla de valores usandolo como indice
                  #prob a partir de la tabla
                  prob = self.tablaValores[idx_atri][idx_clase][value]
                  ##print "\tP(x"+str(idx_atri)+"="+str(value)+"|clase"+str(idx_clase)+")="+str(prob)
                  ##print "\tlog(P(x" + str(idx_atri) + "=" + str(value) + "|clase" + str(idx_clase) + ")=" + str(math.log(prob))
                  #print '\tprobDiscreta:',prob   
              else: #caso continuo
                  value = tupla[idx_atri]
                  prob = self.normpdf(value, self.tablaMedia[idx_atri][idx_clase], self.tablaStd[idx_atri][idx_clase])
                  #prob = scipy.stats.norm(self.tablaMedia[idx_atri][idx_clase], self.tablaStd[idx_atri][idx_clase]).pdf(value)
              #check para descartar el calculo + que no pete con los log
              if prob != 0.0:
                 sumatorio += math.log(prob)
          if probClase != 0.0:
                  sumatorio += math.log(probClase)
          arg.append(sumatorio)
          #print '\tProb NP para esa clase',idx_clase,', quitando logs=',math.exp(sumatorio)
      #return max(arg)
      index, element = max(enumerate(arg), key=itemgetter(1))
      #print 'index:',index,'element:',element
      #print 'arg:',arg
      #print 'indice max(arg):',index
      return index

  # Calcula la probabilidad a priori P(nombreColumna=clase)
  @staticmethod
  def probAPriori(datos, diccionarios, idxColumna, clase):

      numFilas = datos.shape[0]
      # Obtener el valor del diccionario para esa clase
      idClase = diccionarios[idxColumna][clase]
      # Contar las ocurrencias para ese valor del diccionar en esa columna
      numOcurrencias = Counter(datos[:, idxColumna])[idClase]
      return numOcurrencias / numFilas

  @staticmethod
  def probMaxVerosimil(diccionarios, datos, idxAtributo, atributo, idxClass, dominio):
      # =>print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
      # =>prob3 = clasificador.probMaxVerosimil(dataset, "MLeftSq", "b", "Class", "positive")
      # fetch del indice de 'Class'
      idClase = diccionarios[idxClass][dominio]
      # columna 'Class' en forma de array
      # VERSION ANTERIOR:classColumn = dataset.datos[:, idxClass]
      classColumn = datos[:, idxClass]
      # lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i, colValue in enumerate(classColumn) if colValue == idClase]

      idAtributo = diccionarios[idxAtributo][atributo]
      # VERSION ANTERIOR:atriColumn = dataset.datos[:,idxAtributo]
      atriColumn = datos[:, idxAtributo]
      matchesList = itemgetter(*idxMatchClass)(atriColumn)
      countfilter = Counter(matchesList)[idAtributo]
      return countfilter / len(idxMatchClass)

  @staticmethod
  def probMaxVerosimilLaplace(diccionarios, datos, idxAtributo, atributo, idxClass, dominio):
      # =>print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
      # =>prob3 = clasificador.probMaxVerosimil(dataset, "MLeftSq", "b", "Class", "positive")
      # fetch del indice de 'Class'
      idClase = diccionarios[idxClass][dominio]
      # columna 'Class' en forma de array
      # VERSION ANTERIOR:classColumn = dataset.datos[:, idxClass]
      classColumn = datos[:, idxClass]
      # lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i, colValue in enumerate(classColumn) if colValue == idClase]

      idAtributo = diccionarios[idxAtributo][atributo]
      # VERSION ANTERIOR:atriColumn = dataset.datos[:,idxAtributo]
      atriColumn = datos[:, idxAtributo]
      matchesList = itemgetter(*idxMatchClass)(atriColumn)

      countfilter = Counter(matchesList)[idAtributo]
      # LAPLACe
      # Numero de valores posibles del del atributo
      dic = diccionarios[idxAtributo]
      countfilter += 1
      total = len(idxMatchClass) + len(dic)
      return countfilter / total

  # Calcula la Media y desviación típica de los atributos continuos condiconados
  # a la clase.
  @staticmethod
  def mediaDesviacionAtr(datos, diccionarios, idxColumna, idxColumnaClase, clase):
      # Obtener el valor del diccionario para esa clase
      idClase = diccionarios[idxColumnaClase][clase]

      # Lista de índices donde la clase es la que nos pasan
      indices, = np.where(datos[:, idxColumnaClase] == idClase)
      media = np.mean(datos[indices, idxColumna])
      std = np.std(datos[indices, idxColumna]) + 1e-6  # + 0.000001
      return media, std

  # Cuenta el número de repeticiones del atributo dada la clase
  @staticmethod
  def contarAtributos(datos, diccionarios, idxColumna, idxColumnaClase, clase):

      # Obtener el valor del diccionario para esa clase
      idClase = diccionarios[idxColumnaClase][clase]
      clases = diccionarios[idxColumna]
      # Lista de índices donde la clase es la que nos pasan
      indices, = np.where(datos[:, idxColumnaClase] == idClase)
      # Para el atributo dado recorrer todos sus valore y contarlos
      arrayNum = []
      sorted_x = sorted(clases.items(), key=operator.itemgetter(1))
      for key, value in sorted_x:
          # idCl =  diccionarios[idxColumna][cl]
          numOcurrencias = Counter(datos[indices, idxColumna])[value]
          arrayNum.append(numOcurrencias)
      return arrayNum

      # Realiza una clasificacion utilizando una estrategia de particionado determinada.
      # para los apartados
  @staticmethod
  def validacionApartado(particionado, dataset, clasificador, numApartado):

      particiones = particionado.creaParticiones(dataset.datos)

      if particionado.nombreEstrategia == "ValidacionSimple":
          print "Indices train y test para [" + str(particionado.numeroParticiones) + "] particiones:"
      elif particionado.nombreEstrategia == "ValidacionCruzada":
          print 'Datos de train y test para [', particionado.numeroParticiones, '] grupos:'
      else:
          print "ERR: nombre de estrategia no valido"
          exit(1)

      print 'Apartado num:', numApartado
      # for each particion: clasificar y sacar los errores de cada evaluación
      for idx, p in enumerate(particiones):
          if numApartado == 3:
              datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
              # Prob. a priori
              idxColumna = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probAPriori(datosTrain, dataset.diccionarios, idxColumna, "positive")
              print "Prob. a priori para P(Class=positive)", prob

              prob = ClasificadorNaiveBayes.probAPriori(datosTrain, dataset.diccionarios, idxColumna, "negative")
              print "Prob. a priori para P(Class=negative)", prob

              idxAtributo = dataset.nombreAtributos.index("MLeftSq")
              idxClass = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "b", idxClass,
                                                   "positive")
              print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)", prob

              idxAtributo = dataset.nombreAtributos.index("TRightSq")
              idxClass = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "x", idxClass,
                                                   "negative")
              print "Prob. de máxima verosimilitud para P(TRightSq=x|Class=negative)", prob

              idxAtributo = dataset.nombreAtributos.index("MLeftSq")
              idxClass = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probMaxVerosimilLaplace(dataset.diccionarios, datosTrain, idxAtributo, "b",
                                                          idxClass, "positive")
              print "Prob. de máxima verosimilitud con corrección de Laplace para P(MLeftSq=b|Class=positive)", prob

              idxAtributo = dataset.nombreAtributos.index("TRightSq")
              idxClass = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probMaxVerosimilLaplace(dataset.diccionarios, datosTrain, idxAtributo, "x",
                                                          idxClass, "negative")
              print "Prob. de máxima verosimilitud con corrección de Laplace para P(TRightSq=x|Class=negative)", prob

              return

          elif numApartado == 4:
              datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
              # Prob. a priori
              idxClase = dataset.nombreAtributos.index("Class")
              prob = ClasificadorNaiveBayes.probAPriori(datosTrain, dataset.diccionarios, idxClase, "+")
              print "Prob. a priori para P(Class=positive)", prob

              prob = ClasificadorNaiveBayes.probAPriori(datosTrain, dataset.diccionarios, idxClase, "-")
              print "Prob. a priori para P(Class=negative)", prob

              idxAtributo = dataset.nombreAtributos.index("A7")
              prob = ClasificadorNaiveBayes.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "bb", idxClase,
                                                   "+")
              print "Prob. de máxima verosimilitud para P(A7=bb|Class=+)", prob

              idxAtributo = dataset.nombreAtributos.index("A4")
              prob = ClasificadorNaiveBayes.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "u", idxClase,
                                                   "-")
              print "Prob. de máxima verosimilitud para P(A4=u|Class=-)", prob

              idxAtributo = dataset.nombreAtributos.index("A2")
              media, std = ClasificadorNaiveBayes.mediaDesviacionAtr(datosTrain, dataset.diccionarios, idxAtributo, idxClase,
                                                           "+")
              print "Media (", media, ") y desviación típica (", std, ") del atributo A2 condicionado a clase +"

              idxAtributo = dataset.nombreAtributos.index("A14")
              media, std = ClasificadorNaiveBayes.mediaDesviacionAtr(datosTrain, dataset.diccionarios, idxAtributo, idxClase,
                                                           "+")
              print "Media (", media, ") y desviación típica (", std, ") del atributo A14 condicionado a clase +"

              idxAtributo = dataset.nombreAtributos.index("A15")
              media, std = ClasificadorNaiveBayes.mediaDesviacionAtr(datosTrain, dataset.diccionarios, idxAtributo, idxClase,
                                                           "+")
              print "Media (", media, ") y desviación típica (", std, ") del atributo A15 condicionado a clase +"
              return
          else:
              print "Número de apartado (", numApartado, ") incorrecto. Por favor introduzca 3 o 4"
              return

##############################################################################
class ClasificadorMulticlase(Clasificador):
    def __init__(self, clasificadorbase):
        self.clasificadorbase = clasificadorbase
        self.clasificadores = []

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):
        # se van diferentes labels en funcion de la estrategia multiclase
        n_classes = len(diccionario[-1])
        self.clasificadores = []
        ovadiccionario = deepcopy(diccionario)
        ovadiccionario[-1] = {'-': 0, '+': 1}

        for i in range(n_classes):
            new_y = np.zeros((datostrain.shape[0], 1))
            new_y[datostrain[:, -1] == i, :] = 1
            self.clasificadores.append(deepcopy(self.clasificadorbase))
            self.clasificadores[i].entrenamiento(np.append(datostrain[:, :-1], new_y, axis=1), atributosDiscretos, ovadiccionario)

    def clasifica(self, datostest, atributosDiscretos, diccionario, correccion=False):

        scores = np.zeros((datostest.shape[0], len(self.clasificadores)))
        ovadiccionario = deepcopy(diccionario)
        ovadiccionario[-1] = {'-': 0, '+': 1}

        # evaluar el score para cada clasificador one-versus-all
        for i, c in enumerate(self.clasificadores):
            scores[:, i] = c.score(datostest, atributosDiscretos, ovadiccionario)[:, 1]

        # se predice como aquella clase con mas confianza
        preds = np.argmax(scores, axis=1)
        return preds
  