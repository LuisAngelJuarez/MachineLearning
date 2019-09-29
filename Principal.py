import  ArreglaDatos
import numpy as np
import redNeuronal
import pylab as plt
import math

[P,T]=ArreglaDatos.run();
Red1=redNeuronal.redNeuronal(np.array([13,6,3]));       #inicializa la red neuronal con valores aleatorios
[p,t] = ArreglaDatos.get_dato(P,T,5)


error=[]
for j in range(100):
    e=0;
    etha=0.5
    for i in range(0,T.shape[0]):
        [p,t] = ArreglaDatos.get_dato(P,T,i)
        e+=Red1.Entrena_red(p,t,etha);
    e=e/T.shape[0]
    error.append(e)
[p,t] = ArreglaDatos.get_dato(P,T,8)
[out,outs,der]=Red1.calcula_salida(p)
print(t)
print(out)
plt.plot(error)
plt.show()