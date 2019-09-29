import numpy as np
class redNeuronal:
    def __init__(self, a): #a contiene el vector de numero de nodos por capa, incluyendo la salida y entrada
        self.Num_capas=a.shape[0];
        self.Num_neuronas=a;
        self.W=[];# lista vacia para guardar las matrices w
        self.B=[];#lista vacia para guardar los bias
        for i in range(1,self.Num_capas) :
            self.W.append(np.random.rand(self.Num_neuronas[i],self.Num_neuronas[i-1])*0.2-0.1)
            self.B.append(np.random.rand(self.Num_neuronas[i],1)*0.2-0.1)
    def calcula_salida(self,input):     #Calcula la salida de la red neuronal
        Outputs=[]
        derivada=[]
        output=np.copy(input)
        for i in range(0,self.Num_capas-1):
            output=np.dot(self.W[i],output)+self.B[i];
            output=self.Sigmoid(output);
            Outputs.append(output);
            derivada.append(output*(1-output))
        return [np.asarray(output),Outputs,derivada]

    def Sigmoid(self,input): #Genera la función de activación SIGMOIDAL
        output=np.copy(input);
        auxiliar=(1+np.exp(-output))
        output=1/auxiliar
        return output

    def Entrena_red(self,input, T,etha):                            #recibe entrada y target, factor de aprendizaje
        [Output,Outputs,derivada]=self.calcula_salida(input)               #genera la salida de la red neuronal
        delta=[];
        for i in range(len(Outputs)-1,-1,-1):
            j=len(Outputs)-1-i;
            if i==len(Outputs)-1:
                delta.append(np.copy((T-Output)*derivada[i]));
            else:
                delta.append(np.dot(self.W[i+1].transpose(),delta[j-1]*derivada[i+1]))
        for i in range(0,len(delta)):
            if i==0:
                self.B[i]+=etha*delta[len(delta)-1]
                self.W[i]+=etha*delta[len(delta)-1]*input.transpose()
            else:
                self.B[i] += etha*delta[len(delta)-1-i]
                self.W[i] += etha*delta[len(delta)-1-i] * Outputs[i-1].transpose()
        [Out, Outs, der] = self.calcula_salida(input)
        error=np.sum(np.power(Out-T,2))
        return error