import numpy as np

class improve_method:
    def __init__(self):
        pass
    
    def improve(self,weight,E):
        pass

class ADAM(improve_method):
    epsilon,alpha=10**(-8),0.001
    beta_1,beta_2=0.9,0.999
    
    def __init__(self):
        self.m,self.v,self.time=None,None,0
        
    def update(self,weight,E):
        self.time+=1
        
        if(self.time==1):
            self.m,self.v=np.zeros(E.shape),np.zeros(E.shape)
        
        for i in range(E.shape[0]):
            self.m[i]=beta_1*self.m[i]+(1-beta_1)*E[i]
            self.v[i]=beta_2*self.v[i]+(1-beta_2)*(E[i]**2)
        
        m_,v_=m/(1-beta_1**self.times),v/(1-beta_2**self.times)
        weight-=alpha*m_/(np.sqrt(v_)+epsilon)
        
        return weight
    
class GSD(improve_method):
    def __init__(self,rate):
        self.rate=rate
    def update(self,weight,E):
        weight-=self.rate*E
        return weight