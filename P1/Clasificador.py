# -*- coding: utf-8 -*-
from __future__ import division #Para divisiones float por defecto
from operator import itemgetter
from abc import ABCMeta,abstractmethod
from collections import Counter
import numpy as np
import copy
import math
import operator
#import scipy.stats


class Clasificador(object):
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  #Calcula la probabilidad a priori P(nombreColumna=clase)
  @staticmethod
  def probAPriori(dataset, nombreColumna, clase):
      datos = dataset.datos
      numFilas = datos.shape[0]
      #Obtener el índice de la columna deseada
      idxColumna =  dataset.nombreAtributos.index(nombreColumna)
      #Obtener el valor del diccionario para esa clase
      idClase =  dataset.diccionarios[idxColumna][clase]
      #Contar las ocurrencias para ese valor del diccionar en esa columna
      numOcurrencias = Counter(datos[:,idxColumna])[idClase]          
      return numOcurrencias / numFilas

  #Calcula la probabilidad a priori P(nombreColumna=clase)
  @staticmethod
  def probAPriori2(datos, diccionarios, idxColumna, clase):

      numFilas = datos.shape[0]
      #Obtener el valor del diccionario para esa clase
      idClase =  diccionarios[idxColumna][clase]
      #Contar las ocurrencias para ese valor del diccionar en esa columna
      numOcurrencias = Counter(datos[:,idxColumna])[idClase]          
      return numOcurrencias / numFilas

  @staticmethod
  def probMaxVerosimil(diccionarios, datos, idxAtributo, atributo, idxClass, dominio):
      # =>print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)"
      # =>prob3 = clasificador.probMaxVerosimil(dataset, "MLeftSq", "b", "Class", "positive")
      # fetch del indice de 'Class'
      idClase = diccionarios[idxClass][dominio]
      # columna 'Class' en forma de array
      #VERSION ANTERIOR:classColumn = dataset.datos[:, idxClass]
      classColumn = datos[:, idxClass]
      # lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i, colValue in enumerate(classColumn) if colValue == idClase]

      idAtributo = diccionarios[idxAtributo][atributo]
      #VERSION ANTERIOR:atriColumn = dataset.datos[:,idxAtributo]
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
      #VERSION ANTERIOR:classColumn = dataset.datos[:, idxClass]
      classColumn = datos[:, idxClass]
      # lista con los indices de las rows que hacen match con esa idClase
      idxMatchClass = [i for i, colValue in enumerate(classColumn) if colValue == idClase]

      idAtributo = diccionarios[idxAtributo][atributo]
      #VERSION ANTERIOR:atriColumn = dataset.datos[:,idxAtributo]
      atriColumn = datos[:, idxAtributo]
      matchesList = itemgetter(*idxMatchClass)(atriColumn)
      
      countfilter = Counter(matchesList)[idAtributo]
      #LAPLACe
      #Numero de valores posibles del del atributo
      dic = diccionarios[idxAtributo]
      countfilter = countfilter + 1
      total = len(idxMatchClass) + len(dic)
      return countfilter / total
  
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
            
  #Cuenta el número de repeticiones del atributo dada la clase
  @staticmethod
  def contarAtributos(datos, diccionarios, idxColumna, idxColumnaClase, clase):  
      
      #Obtener el valor del diccionario para esa clase
      idClase =  diccionarios[idxColumnaClase][clase]
      clases = diccionarios[idxColumna]     
      #Lista de índices donde la clase es la que nos pasan
      indices, = np.where(datos[:,idxColumnaClase] == idClase)
      #Para el atributo dado recorrer todos sus valore y contarlos
      arrayNum = []
      sorted_x = sorted(clases.items(), key=operator.itemgetter(1))
      for key,value in sorted_x:
          #idCl =  diccionarios[idxColumna][cl]
          numOcurrencias = Counter(datos[indices,idxColumna])[value]
          arrayNum.append(numOcurrencias)      
      return arrayNum            
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario,correcion):
    pass
  
  
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

  # Realiza una clasificacion utilizando una estrategia de particionado determinada. 
  # para los apartados
  @staticmethod
  def validacionApartado(particionado,dataset,clasificador,numApartado):
       
       particiones = particionado.creaParticiones(dataset.datos)
       
       if particionado.nombreEstrategia == "ValidacionSimple":
           print "Indices train y test para [" + str(particionado.numeroParticiones) + "] particiones:"
       elif particionado.nombreEstrategia == "ValidacionCruzada":
           print 'Datos de train y test para [', particionado.numeroParticiones,'] grupos:'
       else:
           print "ERR: nombre de estrategia no valido"
           exit(1)

       print 'Apartado num:',numApartado
       #for each particion: clasificar y sacar los errores de cada evaluación
       for idx, p in enumerate(particiones):
           if numApartado == 3:
               datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
               #Prob. a priori
               idxColumna =  dataset.nombreAtributos.index("Class")               
               prob = Clasificador.probAPriori2(datosTrain, dataset.diccionarios, idxColumna, "positive")
               print "Prob. a priori para P(Class=positive)",prob

               prob = Clasificador.probAPriori2(datosTrain, dataset.diccionarios, idxColumna, "negative")
               print "Prob. a priori para P(Class=negative)",prob

               idxAtributo =  dataset.nombreAtributos.index("MLeftSq")
               idxClass =  dataset.nombreAtributos.index("Class") 
               prob = Clasificador.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "b", idxClass, "positive")
               print "Prob. de máxima verosimilitud para P(MLeftSq=b|Class=positive)",prob                
               

               idxAtributo =  dataset.nombreAtributos.index("TRightSq")
               idxClass =  dataset.nombreAtributos.index("Class") 
               prob = Clasificador.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "x", idxClass, "negative")
               print "Prob. de máxima verosimilitud para P(TRightSq=x|Class=negative)",prob                


               idxAtributo =  dataset.nombreAtributos.index("MLeftSq")
               idxClass =  dataset.nombreAtributos.index("Class") 
               prob = Clasificador.probMaxVerosimilLaplace(dataset.diccionarios, datosTrain, idxAtributo, "b", idxClass, "positive")
               print "Prob. de máxima verosimilitud con corrección de Laplace para P(MLeftSq=b|Class=positive)",prob                
               

               idxAtributo =  dataset.nombreAtributos.index("TRightSq")
               idxClass =  dataset.nombreAtributos.index("Class") 
               prob = Clasificador.probMaxVerosimilLaplace(dataset.diccionarios, datosTrain, idxAtributo, "x", idxClass, "negative")
               print "Prob. de máxima verosimilitud con corrección de Laplace para P(TRightSq=x|Class=negative)",prob                 
               
               return
               
           elif numApartado == 4:
               datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
               #Prob. a priori
               idxClase =  dataset.nombreAtributos.index("Class")               
               prob = Clasificador.probAPriori2(datosTrain, dataset.diccionarios, idxClase, "+")
               print "Prob. a priori para P(Class=positive)",prob

               prob = Clasificador.probAPriori2(datosTrain, dataset.diccionarios, idxClase, "-")
               print "Prob. a priori para P(Class=negative)",prob
               
               idxAtributo =  dataset.nombreAtributos.index("A7")
               prob = Clasificador.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "bb", idxClase, "+")
               print "Prob. de máxima verosimilitud para P(A7=bb|Class=+)",prob
               
               idxAtributo =  dataset.nombreAtributos.index("A4")
               prob = Clasificador.probMaxVerosimil(dataset.diccionarios, datosTrain, idxAtributo, "u", idxClase, "-")
               print "Prob. de máxima verosimilitud para P(A4=u|Class=-)",prob
               
               idxAtributo =  dataset.nombreAtributos.index("A2")
               media,std = Clasificador.mediaDesviacionAtr2(datosTrain, dataset.diccionarios, idxAtributo, idxClase, "+") 
               print "Media (",media,") y desviación típica (",std,") del atributo A2 condicionado a clase +"    

               idxAtributo =  dataset.nombreAtributos.index("A14")
               media,std = Clasificador.mediaDesviacionAtr2(datosTrain, dataset.diccionarios, idxAtributo, idxClase, "+") 
               print "Media (",media,") y desviación típica (",std,") del atributo A14 condicionado a clase +"

               idxAtributo =  dataset.nombreAtributos.index("A15")
               media,std = Clasificador.mediaDesviacionAtr2(datosTrain, dataset.diccionarios, idxAtributo, idxClase, "+") 
               print "Media (",media,") y desviación típica (",std,") del atributo A15 condicionado a clase +"                
               return
           else:
               print "Número de apartado (",numApartado,") incorrecto. Por favor introduzca 3 o 4"
               return

    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  @staticmethod
  def validacion(particionado,dataset,clasificador,correcionL=False,seed=None):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
      
    
       particiones = particionado.creaParticiones(dataset.datos)
       arrayErrores = np.empty(particionado.numeroParticiones)
       if particionado.nombreEstrategia == "ValidacionSimple":
           print "Indices train y test para [" + str(particionado.numeroParticiones) + "] particiones:"
       elif particionado.nombreEstrategia == "ValidacionCruzada":
           print 'Datos de train y test para [', particionado.numeroParticiones,'] grupos:'
       else:
           print "ERR: nombre de estrategia no valido"
           exit(1)

       print 'Correción de Laplace:',correcionL    
       #for each particion: clasificar y sacar los errores de cada evaluación
       for idx, p in enumerate(particiones):
           #print "======================================================"
           #print "PARTICION (" + str(idx) + "):"
           #print "======================================================"
           #print p
           datosTrain, datosTest = dataset.extraeDatos([p.indicesTrain, p.indicesTest])
           #print ' =>DatosTrain [', idx, ']:'
           #print datosTrain
           #print ' =>DatosTest [', idx, ']:'
           #print datosTest

           # Entrenamiento
           """print '|||||||||||||||||||||||||||PARTICION ',idx,'|||||||||||||||||||||||||||'
           print 'datosTrain:\n',datosTrain
           print 'datosTest:\n', datosTest"""
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
         probP = self.probAPriori2(datostrain, diccionario, idxColumnaClase, key)
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
                 media, std = self.mediaDesviacionAtr2(datostrain, diccionario, idx, idxColumnaClase, key)
                 arrayM.append(media)
                 arrayS.append(std)
             tablaM[idx] = arrayM
             tablaS[idx] = arrayS
     
     self.tablaValores = tabla
     self.arrayPriori = arrayP
     self.tablaMedia = tablaM
     self.tablaStd = tablaS
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
  def clasifica(self,datostest,atributosDiscretos,diccionario, correcionL):
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
          flag_0 = False
          sumatorio = 0.0
          probClase = self.arrayPriori[idx_clase]
          for idx_atri, atri in enumerate(self.tablaValores):
              #caso discreto
              prob = 0.0
              if atributosDiscretos[idx_atri]:
                  #sacar valor del atributo de test y pasarlo a indice (int)
                  value = int(round(tupla[idx_atri]))
                  #print 'value:',value
                  #hacer el match con la tabla de valores usandolo como indice
                  #prob a partir de la tabla
                  prob = self.tablaValores[idx_atri][idx_clase][value]
                  ##print "\tP(x"+str(idx_atri)+"="+str(value)+"|clase"+str(idx_clase)+")="+str(prob)
                  ##print "\tlog(P(x" + str(idx_atri) + "=" + str(value) + "|clase" + str(idx_clase) + ")=" + str(math.log(prob))
                  #print '\tprobDiscreta:',prob
              #caso continuo
              else:
                  value = tupla[idx_atri]
                  prob = self.normpdf(value, self.tablaMedia[idx_atri][idx_clase], self.tablaStd[idx_atri][idx_clase])
                  #print '\tprobContinua:', prob
              #check para descartar el calculo + que no pete con los log
              if (prob == 0.0) or (probClase == 0.0) or (flag_0 == True):
                  # P(xj|ci)=0
                  #flag_0 = True
                  #break
                  if probClase != 0.0:
                      sumatorio += math.log(probClase)
                  if prob != 0.0:
                      sumatorio += math.log(prob)
              else:
                  sumatorio += math.log(prob)
                  sumatorio += math.log(probClase)
                  ##print '\tsumatorio=',sumatorio
          if (flag_0 == True):
              arg.append(0)
              #arg.append(sumatorio)
          else:
              probClase = self.arrayPriori[idx_clase]
              #print "\tP(clase"+str(idx_clase)+")="+str(probClase)
              #print "\tlog(P(clase"+str(idx_clase)+")="+str(math.log(probClase))
              #print '\tprobClase [',idx_clase,']:', probClase
              if probClase == 0.0:
                  arg.append(0)
               #else:
                  #sumatorio += math.log(probClase)
                  ##print '\tlog(sumatorio_final + log(P(clase):', sumatorio
              arg.append(sumatorio)
              #print '\tProb NP para esa clase, quitando logs=',math.exp(sumatorio)
      #return max(arg)
      index, element = max(enumerate(arg), key=itemgetter(1))
      #print 'index:',index,'element:',element
      #print 'arg:',arg
      #print 'indice max(arg):',index
      return index
    





  