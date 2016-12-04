# -*- coding: utf-8 -*-
#========================================================================================================
#BRIEF: Script de limpieza previo al push + lista TODO + anotaciones
#========================================================================================================

from path import path

#limpiar .pyc
d = path('/home/arave/Documents/REPOS/faa/P2')
files = d.walkfiles("*.pyc")
for f in files:
    f.remove()
    print "Removed {} file".format(f)
"""
#limpiar plots
d = path('/home/arave/Documents/REPOS/faa/P2/PlotsGenerados')
files = d.walkfiles("*.png")
for f in files:
    f.remove()
    print "Removed {} file".format(f)
"""
"""
========================================================================================================
INFO:
========================================================================================================
- implementado mecanismo dinámico para cambiar parametros sin redefinir funciones/guarreos
    - el diccionario 'mode' recoge una serie de atributos que varian de valor entre 'devault' y <valor>
      que alteran al algoritmo. Ver comentarios en el test.

- corregido cruce
    - desbalanceo
    - ahora con el argumento randSon se reparten las reglas sobrantes de los progenitores entre los
      hijos de manera random.

- implementada mutacion
    - se selecciona un 10% de la poblacion como candidatos y se evalua para cada indvidiuo una prob.
      del 0,1%. Si se cumple, se cambia un bit aleatorio de la primera regla.

- unificado y balanceado el algoritmo para que se ajusten las proporciones de elitismo, cruce, mutacion
  y fill de manera correcta. Tambien he introducido comprobaciones para que no rompa con poblaciones
  de testeo o quede una poblacion nueva mas pequeña de lo normal.

- arreglados problemas de referenciación que hacía que alteraciones de individuos en newPoblacion
  afectara a poblacion. Si el deepcopy del final afecta mucho al rendimiento, se puede refinar
  (conjuntos de indices).

- reestructuracion (siguiendo un poco la correcion), depuracion en general.

========================================================================================================
TO-DO:
========================================================================================================

- validacion cruzada en 2 puntos, si se quiere probar
- test con ficheros tochos y mas generaciones


========================================================================================================
NOTAS:
========================================================================================================

MEME-INSTALATOR para modulos python (probar siempre primero):
	sudo easy_install <nombre_modulo>.py

Lanzar el servidor de jupyter notebook:
	arave@arave-XMG:/opt/anaconda2/bin$ ./jupyter notebook --notebook-dir=/home/arave/Documents/REPOS/faa/P2

Hotkeys notebook:
	Enter Command Mode - ESC
		Insert cell above - A
		Insert cell below - B
		Delete cell		  - D D
		
	Enter Edit Mode	 - ENTER
		Run cell, select below - Shift+Enter
		Run selected cells	   - Ctrl+Enter

Sacar imagenes por jupyter notebook:
    from IPython.display import Image
    Image(filename="example.png")

las tablas en Markdown son #@!*>:(
    http://meta.stackexchange.com/questions/73566/is-there-markdown-to-create-tables

Dar formato con markdown:
	https://support.zendesk.com/hc/en-us/articles/203691016-Formatting-text-with-Markdwn
	

	...                                                 
	array([ 0.97...,  0.93...,  0.95...])
"""
