class NeuralNetWork:
    """do not include output_layer in construct \n
    
        (1) set(add) layers
        (2) 
    """
    
    def __init__(self,*layers,improvement):
        """
        Args
        ____
            layers include input and output layers \n
            improvement is class
        ____
        
        """
        self.layers=[]
        self.add_layers(layers)
        self.weights=[None for w in range(len(layers)-1)]
        self.biases=[None for w in range(len(layers)-1)]
        self.improvements=[improvement() for w in range(len(layers)-1)]
        
        #self.weight=np.random.randn(unit_size,input_size)*(math.sqrt(2/input_size))
        #self.bias=np.zeros(unit_size)*0.01
        
    def add_layers(self,layers):
        """
        Args
        ______
            layers (Layer list) 
        ______
        
        warn!: you should pass list of layers \n
        add layers into Network's layer list
        """
        
        for w in layers:
            self.add_layer(w)
        
    def set_data(self,teachers,variables,norm=False):
        
        """if you want to use normalize date ,norm=True\n
        it doesn't normalize teacher date"""
        
        self.norm=norm
        self.teachers=teachers
        self.variables=variables
        
        if (self.norm):
            self.mean=variables.mean(axis=0)
            self.std=variables.std(axis=0)
            self.variables=((self.variables-self.mean)/self.std)
    
    def set_improvement(self,improvement):
        
        """
        improvement's class ex(ADAM,GSD)
        """
        self.improvements=[improvement() for w in range(len(layers)-1)]
        
    def forward_calculate(self,index):
        """make output of index-th layer and input of index+1-th layer"""
        
        self.layers[index].generate_output()
        if(index<len(self.layers)-1):
            self.layers[index+1].set_input(np.dot(self.weights[index],self.layers[index].output.T).T+self.bias)
        
    def train(self,epoch,batch_size,nor_loc=None):
        
        start=time()
        count=int(self.variables.shape[0]/batch_size)
        # self.Vs[0] is always None so that this is input layer 
        #
        for w in range(epoch):
            for i in range(count):
                self.layers[0].set_input(
                    self.variables.iloc[batch_size*i:batch_size*(i+1)].as_matrix()
                )
                for p in range(len(self.layers)):
                    forward_calculate(p)
                self.back_propagation(self.teachers.iloc[batch_size*i:batch_size*(i+1)].as_matrix())
            self.shuffle()
            print("roop",w+1)
                
        print("time to finish learning=",time()-start)
        
    def back_propagation(self,teachers):
        if(self.layers[-1].check_input):
            raise Error("ungenerated output_layers's output")
        
        delta=self.layers[-1].output-teachers
        for w in reversed(range(len(self.weights))):
            E=self.back(w,delta)
            self.weights[w]=self.improvements[w].update(self.weights[w],E)
            if(w!=0):
                delta=self.back_delta(w-1,delta)
        
    def back_E(self,index,delta_next):
        """
        delta_next :(layer_size:batch_size)
        """
        E=0
        output=self.layers[index].output
        for w in range(output.shape[0]):
            E+=numpy.tensordot(delta_next[w],output,0)
        E/=output.shape[0]
        return E
    
    def back_delta(self,index,next_delta):
        delta=self.layers[index].dactivater(self.layers[index].output)*np.dot(delta_next,self.weights[index])
        return delta
        
    def shuffle(self):
        
        next_index=self.teachers.index.tolist()
        np.random.shuffle(next_index)
        self.variables.index,self.teachers.index=next_index,next_index
        self.variables,self.teachers=self.variables.sort_index(),self.teachers.sort_index()
            
class Error(Exception):
    def __init__(self,message):
        print(message)