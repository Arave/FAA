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
TO-DO:
========================================================================================================

ALL ^^


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
