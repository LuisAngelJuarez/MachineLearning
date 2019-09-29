import numpy as np

def run():
    datos = np.loadtxt('wine.data', dtype=float, delimiter=',');
    input=np.zeros((datos.shape[0],13))
    T=np.zeros([datos.shape[0],3])
    index=np.arange(datos.shape[0]).copy()
    np.random.shuffle(index)
    i=0;
    for j in index:
        if datos[j,0]==1:
            T[i,:]=np.array([0,0,1])
        if datos[j, 0] == 2:
            T[i, :] = np.array([0,1,0])
        if datos[j, 0] == 3:
            T[i, :] = np.array([1,0,0])
        input[i]=datos[j,1:14]
        i+=1;

    for i in range(0,13):
        input[:,i]=input[:,i]/np.amax(input[:,i])
    return [input,T]

def get_dato(input,T,i):
    t=T[i,:].copy();
    p=input[i,:].copy();
    p=np.reshape(p,(13,1))
    t=np.reshape(t,(3,1))
    return [p,t]