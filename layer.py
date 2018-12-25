
class layer:
    """
    this class consider as j-th layer in below methods \n
    E mean 
    """
    
    def __init__(self,func=lambda x:x,dfunc=lambda x:1):
        """acts=[activater,dactivater]"""
        
        self.activater=func
        self.dactivater=dfunc
        #self.m_w,self.v_w,self.m_b,self.v_b=0,0,0,0
        self.weight,self.bias=0,0
        self.E=0
    
    def set_improvement(self,improve):
        self.improve=improve
    
    def set_size(self,unit_size,input_size):
        """初期値を考慮すること"""
        
        self.weight=np.random.randn(unit_size,input_size)*(1/math.sqrt(input_size))
        self.bias=np.zeros(unit_size)*0.001
        
    def calculate(self,input_Y):
        #self.input=input_Y
        self.V=np.dot(self.weight,input_Y).T+self.bias#hide_size:1
        self.Y=self.activater(self.V)

    def back_E(self,delta_next):
        """
        
        """
        
        #dim=len(self.input)#batch
        self.E=0
        """for i in range(dim):
            a=[self.input[i]*self.delta_next[w] for w in range(len(self.delta_next))]
            self.E+=np.r_[a] 
        self.E/=dim"""
        self.E=self.Y*delta_next
        
    def back_delta(self, delta_next,NextIsNor=False):
        """
        this delta is brought to back layer and used in improvement of its weight
        """
        
        """dim=len(self.V)
        if(not NextIsNor):
            self.delta=self.dactivater(self.V[0])*np.dot(weight_next.T,delta_next)
        else:
            self.delta=delta_next
        """
        self.delta=self.dactivater(self.V)*np.dot(self.weight.T,delta_next)
        
    def back_propagation(self,delta_next,NextIsNor=False):
        """
        whether excute backing delta after update params or before
        """
        self.back_E(delta_next)
        
        #入れ替えも必要か?
        self.weight,self.bias=self.improve.update(self.weight,self.E),self.improve.update(self.bias,self.E)
        self.back_delta(delta_next,NextIsNor)
        
        return self.delta#self.Y
    
class output_layer(layer):

    def __init__(self,func=lambda x:x,dfunc=lambda x:1):
        layer.__init__(self,func=lambda x:x,dfunc=lambda x:1)
        
    def set_size(self,unit_size,input_size):
        self.weight=np.random.randn(unit_size,input_size)*(math.sqrt(2/input_size))
        self.bias=np.zeros(unit_size)*0.01

    def back_delta(self,teacher):
        self.delta=(self.Y-teacher)#.mean(axis=0)

    def back_propagation(self,teacher,input_Y,NextIsNor=False):
        
        self.calculate(input_Y)
        delta=self.back_delta(teacher)
        
        self.back_E()
        self.times+=1
        self.weight,self.m_w,self.v_w=self.improve_adam(
            self.times,self.weight,self.E,self.m_w,self.v_w)
        
        self.bias,self.m_b,self.v_b=self.improve_adam(
            self.times,self.bias,delta,self.m_b,self.v_b)
        
        return delta,self.weight