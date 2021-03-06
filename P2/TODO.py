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
DONE- normalizacion scikit learn
DONE- multiclase scikit learn
DONE- plots (output a carpeta 'PlotsGenerados')
DONE- multiclase de regresion logistica
DONE- multiclase scikit-learn
DONE- revisar .docx de correcion
DONE- revisar resultados tras los cambios del domingo a implementacion score/clasifica

DONE- (80%)- Ipython notebook
WORKING- analisis de resultados


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
	https://support.zendesk.com/hc/en-us/articles/203691016-Formatting-text-with-Markdown
	
Sobre el error que me daba con el algoritmo solver 'newton-cg':
	http://stats.stackexchange.com/questions/184017/how-to-fix-non-convergence-in-logisticregressioncv

Error que daba en consola al utilizar el solver 'newton-cg'/'sag' con max_iter=100-1500:
	python ice default io error handler doing an exit exit()  ... errno = 32

Pipelining de transformadores + estimacion, scikit learn:
	#A Pipeline makes it easier to compose estimators, providing this behavior under cross-validation:
	>>> from sklearn.pipeline import make_pipeline
	>>> clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
	>>> cross_val_score(clf, iris.data, iris.target, cv=cv)
	...                                                 
	array([ 0.97...,  0.93...,  0.95...])
"""
