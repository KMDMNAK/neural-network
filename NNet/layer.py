
class layer:
    
    """
    this class consider as j-th layer in below methods \n
    E mean 
    """
    
    def __init__(self,unit_size,func=lambda x:x,dfunc=lambda x:1):
        """acts=[activater,dactivater]"""
        
        self.activater=func
        self.dactivater=dfunc
        #self.m_w,self.v_w,self.m_b,self.v_b=0,0,0,0
        self.input,self.output=0,0
        self.unit_size=unit_size
        self.input_check=False#inputが更新されてoutputが非更新の時True
    
    def set_input(self,new_input):
        self.input=new_input
        if(self.input.shape[1]!=unit_size):
            raise Error("mismatch columns number of input")
        
        self.input_check=True
        
    def generate_output(self):
        if(not input_check):
            raise Error("update input or turn input_check into True")
        self.output=self.activater(self.input)
        self.input_check=False