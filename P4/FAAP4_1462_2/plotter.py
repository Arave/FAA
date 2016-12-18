import matplotlib.pyplot as plt
import numpy as np
import datetime, time

def genPlot(plotName,arrayFitnessMejores, arrayFitnessMedio):
    plt.figure(figsize=((15,5)))
    plt.hold(True)
    plt.plot(arrayFitnessMejores, 'bo-', label='Mejor')
    plt.plot(arrayFitnessMedio, 'ro-', label='Media')
    plt.legend(loc=4)
    plt.xlabel('Generacion')
    plt.ylabel('Fitness')
    plt.ylim(np.min([arrayFitnessMejores, arrayFitnessMedio])*0.95,
             np.max([arrayFitnessMejores, arrayFitnessMedio])*1.05)

    #plt.savefig('../Graficas/' + plotName + '.png')
    #print "PLOTTING..."
    #plotName = 'Graficas/'+ str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))+'.png'
    #plotName = 'Graficas/1.png'
    plt.savefig('Graficas/'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))+'.png')
    return