class NetWork:
    """do not include output_layer in construct"""
    
    def __init__(self,*layers):
        self.layers,self.OutPut_layer=[],None
        self.add_layers(layers)
        
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
            
    def add_output_layer(self,out_layer):
        self.OutPut_layer=out_layer
        
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
        
    def set_layers_size(self,size_list):
        """warning : you should excute this after seting datas and input and output"""
        
        for w in range(len(self.layers)):
            self.layers[w].set_size(size_list[w+1],size_list[w])
            
        self.OutPut_layer.set_size(self.teachers.shape[1],size_list[-2])
    
    def set_improvement(self,improve):
        """
        improvement's class ex(ADAM,GSD)
        """
        
        for w in self.layers:
            w.set_improvement(improve)
        
    def train(self,epoch,batch_size,nor_loc=None,improve="adam",clear="False"):
        start=time()
        
        if(improve=="adam"):
            count=int(self.variables.shape[0]/batch_size)
            print("count=",count)
            for w in range(epoch):
                for i in range(count):
                    
                    X=self.variables.iloc[batch_size*i:batch_size*(i+1)].as_matrix()#date_size:1
                    Y=self.teachers.iloc[batch_size*i:batch_size*(i+1)].as_matrix()
                    
                    X1=X
                    for p in range(len(self.layers)):
                        X1=self.layers[p].cal(X1)
                    two=self.OutPut_layer.back_propagation(Y,X1)
                    for l in reversed(range(len(self.layers))):
                        check=True if(nor_loc!=None and l+2 in nor_loc) else False
                        two=self.layers[l].back_propagation(two[0],two[1],NextIsNor=check)
                self.shuffle()
                print("roop",w+1)
                
        print("time to finish learning=",time()-start)

    def shuffle(self):
        next_index=self.teachers.index.tolist()
        np.random.shuffle(next_index)
        self.variables.index=next_index
        self.teachers.index=next_index
        self.variables=self.variables.sort_index()
        self.teachers=self.teachers.sort_index()

    def check(self,teachers_test,variables_test,batch_size=1):
        error=np.zeros(len(variables_test))
        count=int(variables_test.shape[0]/batch_size)
        for i in range(count):
            
            X=variables_test.iloc[batch_size*i:batch_size*(i+1)]#date_size:1
            Y=teachers_test.iloc[batch_size*i:batch_size*(i+1)].as_matrix()
            if(self.norm):
                X=(((X-self.mean)/self.std)).as_matrix()
            else:
                X=X.as_matrix()
            for p in range(len(self.Layers)):
                X=self.Layers[p].cal(X)
            X=self.OutPut_layer.cal(X)
            e=(np.abs((X-Y)/Y)*100)
            error[i]=float(e.mean(axis=1).mean(axis=0))
        print("error.shape=",len(error))
        under=np.where(error<100)[0]
        print("rate of error below 100%=",100*len(under)/len(error),"%")
        print("error average percent=",error.mean(),"%")
        print("error average percent under 100%=",error[under].mean(),"%")
        plt.hist(error,bins=100,range=(0,100))
        plt.show()

    def check_number(self,teachers_test,variables_test,show_num):
        error=np.zeros(show_num)
        for i in range(show_num):
            X=variables_test.iloc[i]
            Y=teachers_test.iloc[i]
            
            if(self.norm):
                X=((X.T-self.mean)/self.std).T
            
            for l in range(len(self.Layers)):
                X=self.Layers[l].cal(X)
            X=self.OutPut_layer.cal(X)
            e=(np.abs((X-Y)/Y)*100)
            error[i]=float(e.mean(axis=0))
            
            print("OutPut=",X.tolist())
            print("Teacher=",Y.tolist(),"\n")
        print("error.shape=",len(error))
        under=np.where(error<100)[0]
        print("rate of error below 100%=",100*len(under)/len(error),"%")
        print("error average percent=",error.mean(),"%")
        print("error average percent under 100%=",error[under].mean(),"%")
        
    def check_class(self,teachers_test,variables_test,show_num):
        pass