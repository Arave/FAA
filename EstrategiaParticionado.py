from __future__ import print_function
from abc import ABCMeta,abstractmethod
import numpy as np

class Particion:
  
  indicesTrain=[]
  indicesTest=[]
  
  def __init__(self, indicesTrain, indicesTest):
    self.indicesTrain= indicesTrain
    self.indicesTest= indicesTest

  def __str__(self):
      s = ">iTrain:["
      for i in self.indicesTrain:
          s = s + str(i) + " "
      s += "]\n>itest:["
      for i in self.indicesTest:
          s = s + str(i) + " "
      s += "]\n"
      return s

#####################################################################################################

class EstrategiaParticionado(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0 #Valdicacion cruzada
  particiones=[] #lista de objetos particion
  numParticionesSimples=0 #Validacion simple
  porcentajeParticiones=0 #Validacion simple
  
  
  
  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  

  
  def __init__(self, numParticionesSimples, porcentajeParticiones):
      
      EstrategiaParticionado.nombreEstrategia = "ValidacionSimple"
      EstrategiaParticionado.numParticionesSimples = numParticionesSimples
      EstrategiaParticionado.porcentajeParticiones = porcentajeParticiones
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    
    np.random.seed(seed)                            #semilla, "default: system time to generate next random num"
    indices = np.random.permutation(datos.shape[0]) #random perm del indice de filas de datos
    #Porcentajes
    numTraining = (self.porcentajeParticiones * len(indices)) / 100

    training_idx, test_idx = indices[:numTraining], indices[numTraining:]
    #training, test = datos[training_idx,:], datos[test_idx,:]

    #nueva version con parametro:
    for i in range(self.numParticionesSimples):
        indices = np.random.permutation(datos.shape[0])
        numTraining = (self.porcentajeParticiones * len(indices)) / 100
        training_idx, test_idx = indices[:numTraining], indices[numTraining:]
        self.particiones.append(Particion(training_idx, test_idx))

    return self.particiones

    #Comprobar si hay num de particiones .
    #Si es num
    #Crear particiones simples
    #np.split(datos, self.numeroParticiones)
       
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):

  def __init__(self, numparticiones):
      EstrategiaParticionado.nombreEstrategia = "ValidacionCruzada"
      EstrategiaParticionado.numeroParticiones = numparticiones #k

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):   
    np.random.seed(seed)
    indices = np.random.permutation(datos.shape[0])  # random perm del indice de filas de datos
    grupos = np.array_split(indices, self.numeroParticiones)
    glist = []
    for g in grupos:
        glist.append(g.tolist())  
    for idx,g in enumerate(glist):
        glistneg = glist[:]
        glistneg.pop(idx)
        flattened = [val for sublist in glistneg for val in sublist]
        self.particiones.append(Particion(flattened,g)) #(indicesTrain, indicesTest)
    return self.particiones
    
    
