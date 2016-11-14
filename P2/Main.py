from sklearn import neighbors, linear_model, preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from Datos import Datos

class Main(object):

    @staticmethod
    def genDataTxt(filename, data):
        fhandle = open(filename, 'w')
        fhandle.write("[")
        for row in data:
            fhandle.write("[")
            for item in row:
                fhandle.write("%.1f " % item)
            fhandle.write("]\n")
        fhandle.write("]")
        fhandle.close()
        return

    @staticmethod
    def pruebasMatices():
        dataset = Datos('./ConjuntosDatos/tic-tac-toe.data', True)
        encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
        x = encAtributos.fit_transform(dataset.datos[:, :-1])  # matriz de datos codificada
        # y = dataset.datos[:,-1]                                 #clase de cada patron

        # iris = datasets.load_iris()
        Main.genDataTxt('datos_numpy.txt', dataset.datos)
        Main.genDataTxt('datos_encoded.txt', x)
        # genDataTxt('datos_iris_data.txt', iris.data)
        return

    @staticmethod
    def run(fichero_datos, cls, cls_brief, strat, k, supervisado, laplace, normalizar=False, separador=False):
        if separador:
            print "\n"
            print "==============================================================================="
            print "=============================(", fichero_datos, ")============================="
            print "==============================================================================="
        print "\nFichero de datos: " + fichero_datos
        print "\nLaplace =" + str(laplace) + ", normalizar = ", str(normalizar)
        dataset = Datos('./ConjuntosDatos/' + fichero_datos, supervisado)
        print "Estrategia: ", strat.getStratname(), ", numParticiones: ", str(k)
        print "Clasificador: ", cls_brief
        print "Ejecucion: "
        print "________________________________________________________________"
        plotName = 'PlotsGenerados/' + fichero_datos + '-' + cls_brief + '-' + 'normalizar=' + str(normalizar) + '.png'
        errores = cls.validacion(strat, dataset, cls, laplace, normalizar, plotName=plotName)
        print "\n"
        return errores

    # Documentacion:
    # http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
    # http://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
    # Sobre HybridNB: http://bit.ly/2eVXXcQ
    # Sobre L1 y L2 (regresion logistica):
    #   https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization
    # Doc general sobre regresion logisitca:
    #   http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    @staticmethod
    def runScikit(fichero_datos, cls, cls_brief, strat, cv, supervisado, laplace, normalizar=False, k=None):
        dataset = Datos('./ConjuntosDatos/' + fichero_datos, supervisado)
        x = dataset.datos[:, :-1]
        y = dataset.datos[:, -1]
        clf = None
        pre_clf = None
        if cls == "MultinomialNB":
            if laplace:
                alpha = 1.0
            else:
                alpha = 0
            clf = MultinomialNB(alpha, fit_prior=True, class_prior=None)
        elif cls == "GaussianNB":
            clf = GaussianNB()
        elif cls == "HybridNB":
            encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
            x = encAtributos.fit_transform(dataset.datos[:, :-1])  # matriz de datos codificada
            y = dataset.datos[:, -1]  # clase de cada patron
            clf = GaussianNB()
        elif cls == "Prior":
            clf = DummyClassifier(strategy='prior')
        elif cls == "KNeighborsClassifier":
            if normalizar:
                clf = make_pipeline(preprocessing.StandardScaler(),neighbors.KNeighborsClassifier(n_neighbors=k))
            else:
                clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        elif cls == "LogisticRegression":
            pre_clf = linear_model.LogisticRegression(solver='liblinear')
            if normalizar:
                clf = make_pipeline(preprocessing.StandardScaler(), pre_clf)
            else:
                clf = pre_clf
        elif cls == "LRMulticlass":
            #pre_clf = linear_model.LogisticRegression(solver='newton-cg')  # L2 only
            #pre_clf = linear_model.LogisticRegression(solver='sag')        #L2 only, large dataset, may req preproc
            pre_clf = linear_model.LogisticRegression(solver='lbfgs')       #L2 only
            if normalizar:
                #pre_clf = linear_model.LogisticRegression(solver='sag', max_iter=1500)
                clf = make_pipeline(preprocessing.StandardScaler(), pre_clf)
            else:
                clf = pre_clf
        else:
            print "ERR: Clasificador no valido"
            return
        # siempre se aplica cross-validation con cv folds por defecto. Se puede cambiar si piden hold-out
        scores = cross_val_score(clf, x, y, cv=10)
        print "===========RESULTADO Scikit-learn=============="
        print("Overall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        """
        print "Media de errores total:", 100-scores.mean(), "%"
        print "Mediana de errores total:", 100-np.median(scores), "%"
        print "Desviacion tipica:", 100-scores.std(), "%"
        """
        return