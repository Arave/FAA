# -*- coding: utf-8 -*-

from __future__ import division #Para divisiones float por defecto
import numpy as np
import sys

class Datos(object):
  
  supervisado=True
  TiposDeAtributos=('Continuo','Nominal')
  tipoAtributos=[]
  nombreAtributos=[]
  nominalAtributos=[]
  datos=np.array(())
  # Lista de diccionarios. Uno por cada atributo.
  diccionarios=[]
  #Lista de medias de los atributos continuos
  arrayM=[]
  #lista de desviaciones típicas (STDs) de los atributos continúos
  arrayS=[]
  
  def __init__(self, nombreFichero,sup):
      
      
    del self.tipoAtributos[:] #Limpiar la lista de ejecucciones anteriores
    del self.nombreAtributos[:] #Limpiar la lista de ejecucciones anteriores
    del self.nominalAtributos[:] #Limpiar la lista de ejecucciones anteriores
    del self.diccionarios[:] #Limpiar la lista de ejecucciones anteriores      
      
    self.supervisado = sup
    #print ("Tipo:" + str(self.supervisado))
    f = open(nombreFichero,'r')
    line_list = f.readlines()
    line_list[:] = [x.replace('\n', '').replace('\r', '') for x in line_list]

    #Lineas de datos efectivas
    data_lines = len(line_list) - 3

    #nombreDeAtributos
    nombreAtri_list = line_list[1].split(',')
    self.nombreAtributos = nombreAtri_list
    num_atributos = len(nombreAtri_list)

    #tipoDeAtributos, nominalAtributos
    tipos_list = line_list[2].split(',')
    try:
        for idx,tipo in enumerate(tipos_list):
            if tipo=="Continuo":
                self.nominalAtributos.append(False)
            elif tipo =="Nominal":
                self.nominalAtributos.append(True)
            else:
                raise ValueError('ERR:tipo leido no valido. ', tipo, idx, 0)
        if idx+1 != num_atributos:
            raise ValueError('ERR:mismatch en numero de nombres y tipos.', tipo, idx, 1)
    except ValueError as err:
        if err.args[3] == 0:
            print err.args[0], "Leido:", err.args[1], ". Index:" + str(err.args[2])
        else:
            print(err.args[0])
        sys.exit(-1)
    self.tipoAtributos = tipos_list

    #diccionarios
    idx = 0
    sublists = []
    listaDicTemporal = []
    #pasar los datos a matriz traspuesta
    for line in line_list[3:]:
        sublists.append(line.strip().split(','))
    transposed = zip(*sublists)
    #ordenar y filtrar duplicados
    for idx,atribute in enumerate(transposed):
        #transposed[idx] = sorted(set(atribute))
		listaDicTemporal.append(sorted(set(atribute)))
    #construir diccionarios
    for idic,atribute in enumerate(listaDicTemporal):
        d = {}
        if self.nominalAtributos[idic]:
            for val_index,value in enumerate(atribute):
                d[value] = val_index
        self.diccionarios.append(d)
    #(!) - http://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key
    #print(str(self.diccionarios))

    #pasar a values del diccionario, NumPy?
    #transposedcpy = zip(*sublists)
    #print(str(transposedcpy))
    self.datos = np.empty([data_lines,len(self.nombreAtributos)], dtype=float)
    for idx,atribute in enumerate(transposed):
        if self.nominalAtributos[idx]:
            #transposed[idx] = [self.diccionarios[idx][x] for x in atribute]
            for i in range(data_lines):
                da = [self.diccionarios[idx][x] for x in atribute][i]
                if da != '?': #no falta el atributo
                    self.datos[i][idx] = da
        else:
            for i in range(data_lines):
                da = transposed[idx][i]
                if da != '?': #no falta el atributo
                    self.datos[i][idx] = da                
            
    #solo para el printeo. En la estructura se mantiene el float completo
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})            

    #print "Datos (numpy)"
    #print self.datos 
   
  # idx: lista de indices de los patrones a extraer 
  def extraeDatos(self,idx):
    #print "idx[0]",idx[0]
    #print "idx[1]",idx[1]
    #print "Datos train:"
    #print self.datos[idx[0],:]
    #print "Datos test:"
    #print self.datos[idx[1],:]
    training, test = self.datos[idx[0],:], self.datos[idx[1],:]  
    return training, test

    #calculará las medias y desviaciones típicas de cada atributo continuo a partir de los datos de entrenamiento
  def calcularMediasDesv(self,datostrain):
       
     del self.arrayM[:] #Limpiar la lista de ejecucciones anteriores
     del self.arrayS[:] #Limpiar la lista de ejecucciones anteriores  
     #Recorrer los atributos excepto el último (Clase)
     for idxColumna,atr in enumerate(self.nominalAtributos[:-1]):
         if atr == False:  #continuo
             #Calcular la media y std para las clases
             media = np.mean(datostrain[:,idxColumna])
             std = np.std(datostrain[:,idxColumna]) + 1e-6  #+ 0.000001     
             self.arrayM.append(media)
             self.arrayS.append(std)

  def normalizarDatos(self,datos):
     #Recorrer los atributos excepto el último (Clase)
     for idxColumna,atr in enumerate(self.nominalAtributos[:-1]):
         if atr == False:  #continuo
             #Restarle la media y dividirlo por la std
             for idxFila,dat in enumerate(datos[:,idxColumna]):
                 datos[idxFila][idxColumna] = (datos[idxFila][idxColumna] - self.arrayM[idxColumna] ) / self.arrayS[idxColumna]
  