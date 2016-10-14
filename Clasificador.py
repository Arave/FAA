# -*- coding: utf-8 -*-
from __future__ import division #Para divisiones float por defecto
from abc import ABCMeta,abstractmethod
from collections import Counter
import numpy as np


class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  

  #Calcula la probabilidad a priori P(nombreColumna=clase)
  def probAPriori(self, dataset, nombreColumna, clase):
      datos = dataset.datos
      numFilas = datos.shape[0]
      #Obtener el índice de la columna deseada
      idxColumna =  dataset.nombreAtributos.index(nombreColumna)
      #Obtener el valor del diccionario para esa clase
      idClase =  dataset.diccionarios[idxColumna][clase]
      #Contar las ocurrencias para ese valor del diccionar en esa columna
      numOcurrencias = Counter(datos[:,idxColumna])[idClase]          
      return numOcurrencias / numFilas

  
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
           print "Indices de train: "
           print particiones.indicesTrain
           print "Indices de test: "
           print particiones.indicesTest
           datosTrain, datosTest = dataset.extraeDatos([particiones.indicesTrain, particiones.indicesTest])
           print "Datos de train y test: "
           print "=>DatosTrain:"
           print datosTrain
           print "=>DatosTest:"
           print datosTest
                     
           predClass = clasificador.entrenamiento(datosTrain)
           print "Predicción (Clase mayoritaria): "
           print predClass
           pred = clasificador.clasifica(datosTest)
           print "Predicción: "
           print pred
           errores = clasificador.error(datosTest,pred)
           print "Porcentaje de errores (%): "
           print errores         
           
           
       elif particionado.nombreEstrategia == "ValidacionCruzada":
           errorTotal = 0
           arrayErrores = np.empty(particionado.numeroParticiones)
           print 'Datos de train y test para [', particionado.numeroParticiones,'] grupos:'
           for idx,p in enumerate(particiones):
               print "Indices de train: "
               print p.indicesTrain
               print "Indices de test: "
               print p.indicesTest               
               datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
               print ' =>DatosTrain [',idx,']:'
               print datosTrain
               print ' =>DatosTest [', idx, ']:'
               print datosTest
               
               #Entrenamiento
               predClass = clasificador.entrenamiento(datosTrain)
               print "Predicción (Clase mayoritaria): "
               print predClass
               pred = clasificador.clasifica(datosTest)
               print "Predicción: "
               print pred
               
               error = clasificador.error(datosTest,pred)
               arrayErrores[idx] = error
               print "Porcentaje de error (%): "
               print error
               
           #estadística
           print arrayErrores    
           print "Media de errores total: "
           print np.mean(arrayErrores)
           print "Mediana de errores total: "
           print np.median(arrayErrores)           
           print "Desviación típica: "
           print np.std(arrayErrores)     
       else:
           print "nombre de estrategia no valido"
           exit(1)
       # estrategia=ValidacionSimple(10,80) => particionado, arg[0] - numero de particiones. Calcular la media y desv.
      

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

 

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    pass
    
     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    pass

    
    





  