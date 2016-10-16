# -*- coding: utf-8 -*-
from __future__ import division #Para divisiones float por defecto
from operator import itemgetter
from abc import ABCMeta,abstractmethod
from collections import Counter
import numpy as np


class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  #Calcula la probabilidad a priori P(nombreColumna=clase)
  @staticmethod
  def probAPriori(dataset, nombreColumna, clase):
      ##print "Prob. a priori para P(Class=positive)"
      ##prob1 = clasificador.probAPriori(dataset, "Class", "positive")
      datos = dataset.datos
      numFilas = datos.shape[0]
      #Obtener el índice de la columna deseada
      idxColumna =  dataset.nombreAtributos.index(nombreColumna)
      #Obtener el valor del diccionario para esa clase
      idClase =  dataset.diccionarios[idxColumna][clase]
      #Contar las ocurrencias para ese valor del diccionar en esa columna
      numOcurrencias = Counter(datos[:,idxColumna])[idClase]          
      return numOcurrencias / numFilas

  """
  def probMaxVerosimil(dataset, nombreColumna, atributo, nombreDominio, dominio):
      #=>print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
      #=>prob3 = clasificador.probMaxVerosimil(dataset, "MLeftSq", "b", "Class", "positive")
      #palce holder porque algo falla (estoy haciendo pruebas en una versión local reducida en PyCharm)
      datos = dataset.datos
      #fetch del indice de 'Class'
      idxClass = dataset.nombreAtributos.index(nombreDominio)
      idClase  = dataset.diccionarios[idxClass][dominio]
      #columna 'Class' en forma de array
      classColumn = dataset.datos[:,idxClass]
      #lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i,colValue in enumerate(classColumn) if colValue==idClase]

      idxAtributo = dataset.nombreAtributos.index(nombreColumna)
      idAtributo = dataset.diccionarios[idxAtributo][atributo]
      atriColumn = dataset.datos[:,idxAtributo]
      matchesList = itemgetter(*idxMatchClass)(atriColumn)
      countfilter = Counter(matchesList)[idAtributo]

      return countfilter/len(idxMatchClass)
      pass
  """

  @staticmethod
  def probMaxVerosimil(diccionarios, datos, idxAtributo, atributo, idxClass, dominio):
      # =>print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
      # =>prob3 = clasificador.probMaxVerosimil(dataset, "MLeftSq", "b", "Class", "positive")
      # fetch del indice de 'Class'
      idClase = diccionarios[idxClass][dominio]
      # columna 'Class' en forma de array
      classColumn = datos[:, idxClass]
      # lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i, colValue in enumerate(classColumn) if colValue == idClase]
      
      idAtributo = diccionarios[idxAtributo][atributo]
      atriColumn = datos[:, idxAtributo]
      matchesList = itemgetter(*idxMatchClass)(atriColumn)
      countfilter = Counter(matchesList)[idAtributo]
      return countfilter / len(idxMatchClass)
  
  #Calcula la Media y desviación típica de los atributos continuos condiconados
  #a la clase.
  @staticmethod
  def mediaDesviacionAtr(dataset, nombreColumna, nombreColumnaClase, clase):
      datos = dataset.datos
      #Obtener el índice de la columna deseada
      idxColumna =  dataset.nombreAtributos.index(nombreColumna)
      #Obtener el índice de la columna de clase
      idxColumnaClase =  dataset.nombreAtributos.index(nombreColumnaClase)      
      #Obtener el valor del diccionario para esa clase
      idClase =  dataset.diccionarios[idxColumnaClase][clase]
      
      #Lista de índices donde la clase es la que nos pasan
      indices, = np.where(datos[:,idxColumnaClase] == idClase)
      media = np.mean(datos[indices,idxColumna])
      std = np.std(datos[indices,idxColumna]) + 1e-6  #+ 0.000001 
      return media, std
      
  #Calcula la Media y desviación típica de los atributos continuos condiconados
  #a la clase.
  @staticmethod
  def mediaDesviacionAtr2(datos, diccionarios, idxColumna, idxColumnaClase,clase):  
      #Obtener el valor del diccionario para esa clase
      idClase =  diccionarios[idxColumnaClase][clase]
      
      #Lista de índices donde la clase es la que nos pasan
      indices, = np.where(datos[:,idxColumnaClase] == idClase)
      media = np.mean(datos[indices,idxColumna])
      std = np.std(datos[indices,idxColumna]) + 1e-6  #+ 0.000001 
      return media, std      
      
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  def error(self,datos,pred):
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
  def validacion(particionado,dataset,clasificador,seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test      
       particiones = particionado.creaParticiones(dataset.datos)

       if particionado.nombreEstrategia == "ValidacionSimple":
           arrayErrores = np.empty(particionado.numParticionesSimples)

           print "Indices train y test para [" + str(particionado.numParticionesSimples) + "] particiones:"
           for idx,p in enumerate(particiones):
               print ">Particion ("+str(idx)+"):"
               print p
               datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
               print ' =>DatosTrain [', idx, ']:'
               print datosTrain
               print ' =>DatosTest [', idx, ']:'
               print datosTest

               # Entrenamiento
               predClass = clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
               print "Predicción (Clase mayoritaria): "
               print predClass
               pred = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios)
               print "Predicción: "
               print pred

               error = clasificador.error(datosTest, pred)
               arrayErrores[idx] = error
               print "Porcentaje de error (%): "
               print error

       elif particionado.nombreEstrategia == "ValidacionCruzada":
           arrayErrores = np.empty(particionado.numeroParticiones)
           print 'Datos de train y test para [', particionado.numeroParticiones,'] grupos:'
           for idx,p in enumerate(particiones):
               print 'Particion (',idx,'):'
               print p
               datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
               print ' =>DatosTrain [',idx,']:'
               print datosTrain
               print ' =>DatosTest [', idx, ']:'
               print datosTest
               
               #Entrenamiento
               predClass = clasificador.entrenamiento(datosTrain, dataset.nominalAtributos, dataset.diccionarios)
               print "Predicción (Clase mayoritaria): "
               print predClass
               pred = clasificador.clasifica(datosTest, dataset.nominalAtributos, dataset.diccionarios)
               print "Predicción: "
               print pred
               
               error = clasificador.error(datosTest,pred)
               arrayErrores[idx] = error
               print "Porcentaje de error (%): "
               print error
       else:
           print "nombre de estrategia no valido"
           exit(1)
       # estrategia=ValidacionSimple(10,80) => particionado, arg[0] - numero de particiones. Calcular la media y desv.
       
       
       #estadística
       print arrayErrores    
       print "Media de errores total: "
       print np.mean(arrayErrores)
       print "Mediana de errores total: "
       print np.median(arrayErrores)           
       print "Desviación típica: "
       print np.std(arrayErrores)  

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
    
  def clasifica(self,datostest,atributosDiscretos=None,diccionario=None):
    # Asignar la clase mayoritaria a todos los datos
      numFilas = datostest.shape[0]
      datos = np.empty(numFilas)
      datos.fill(self.mayoritaria)
      return datos
 

  
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  tablaValores = []
  arrayPriori = []

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      
     print "entrenamiento Naive"
     numColumnas = datostrain.shape[1]
     idxColumnaClase = numColumnas - 1
     clases = diccionario[idxColumnaClase]
     
         
         
     for i,clase in enumerate(clases):
         #Calcular los a priori
         #probP = probAPriori(....)
         #self.arrayPriori.append(probP) 
         #Recorrer los atributos
         for idx,atr in enumerate(atributosDiscretos):
             if(atr == False): #Continuo         
                 #Calcular la media y std para los atributos continuos => gaussiana
                 media, std = self.mediaDesviacionAtr2(datostrain, diccionario, idx, idxColumnaClase, clase)
                 #gaussiana = gaussiana(media,std)
                 #self.tablaValores[i].append(gaussiana)
                 self.tablaValores[i].append(media)
             else: #nominal/discreto
                 atributo = diccionario[idx][i]             
                 
                 self.tablaValores[i].append(probMaxV)
     print self.tablaValores

                 
                 
                 
    
#     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
      #placeholder
      #probMaxV = self.probMaxVerosimil(diccionario, datostrain, idx, atributo, idxColumnaClase, clase)

      return datostest[:,-1]

    
    





  