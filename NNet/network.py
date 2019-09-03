from time import time
import numpy as np


class NeuralNetWork:
    """do not include output_layer in construct \n

        (1) set(add) layers
        (2) 
    """

    def __init__(self, *layers, imp, output=True):
        """
        Args
        ____
            layers include input and output layers \n
            improvement is class
        ____

        """
        self.layers = []

        self.weights = [None for w in range(len(layers)-1)]
        self.biases = [None for w in range(len(layers)-1)]
        self.improvements = [imp() for w in range(len(layers)-1)]

        self.add_layers(layers, output=output)
        # self.weight=np.random.randn(unit_size,input_size)*(math.sqrt(2/input_size))
        # self.bias=np.zeros(unit_size)*0.01

    def add_layers(self, layers, output=True):
        """
        Args
        ______
            layers (Layer list) 
            output(bool) : whether it include output layer
        ______

        warn!: you should pass list of layers \n
        add layers into Network's layer list
        """

        for w in layers:
            self.layers.append(w)
        if(output):
            self.set_size()

    def set_data(self, teachers, variables, norm=False):
        """if you want to use normalize date ,norm=True\n
        it doesn't normalize teacher date"""

        self.norm = norm
        self.teachers = teachers
        if(len(self.teachers.shape) == 1):
            self.teachers = np.reshape(self.teachers, (len(self.teachers), -1))
        self.variables = variables

        if (self.norm):
            self.mean = variables.mean(axis=0)
            self.std = variables.std(axis=0)
            self.variables = ((self.variables-self.mean)/self.std)

    def set_improvement(self, imp):
        """
        improvement's class ex(ADAM,GSD)
        """
        self.improvements = [imp() for w in range(len(self.layers)-1)]

    def set_size(self):
        layer_length, weights_length = len(self.layers), len(self.weights)

        if(layer_length != weights_length+1):
            raise Error("mismatch layer and weights length")
        for w in range(weights_length):
            self.weights[w] = np.random.randn(self.layers[w+1].unit_size,
                                              self.layers[w].unit_size)*(np.sqrt(2/self.layers[w].unit_size))
            self.biases[w] = np.zeros(self.layers[w+1].unit_size)

    def one_forward(self, index):
        """make output of index-th layer and input of index+1-th layer"""

        self.layers[index].generate_output()
        if(index < len(self.layers)-1):
            self.layers[index+1].set_input(
                np.dot(self.weights[index], self.layers[index].output.T).T+self.biases[index])

    def forward_calculate(self):
        for p in range(len(self.layers)):
            self.one_forward(p)

    def forward_excute(self, X):
        self.layers[0].set_input(X)
        self.forward_calculate()

    def train(self, epoch, batch_size, nor_loc=None):

        start = time()
        count = int(self.variables.shape[0]/batch_size)
        # self.Vs[0] is always None so that this is input layer

        for w in range(epoch):

            for i in range(count):
                self.forward_excute(
                    self.variables[batch_size*i:batch_size*(i+1)])
                self.back_propagation(
                    self.teachers[batch_size*i:batch_size*(i+1)])

            self.shuffle()
            print("roop", w+1)

        print("time to finish learning=", time()-start)

    def back_propagation(self, teachers):
        if(self.layers[-1].input_check):
            raise Error("ungenerated output_layers's output")

        delta = self.layers[-1].output-teachers
        for w in reversed(range(len(self.weights))):
            E = self.back_E(w, delta)
            self.weights[w] = self.improvements[w].update(self.weights[w], E)

            if(w != 0):
                delta = self.back_delta(w, delta)

    def back_E(self, index, delta_next):
        """
        delta_next :(layer_size:batch_size)
        index based on weights!!

        """

        E = 0
        outputs = self.layers[index].output
        for output in outputs:
            for w in range(outputs.shape[0]):
                E += np.tensordot(delta_next[w], output, 0)
        E /= output.shape[0]
        return E

    def back_delta(self, index, delta_next):
        """index based on layer!!!!!"""
        delta = self.layers[index].dactivater(
            self.layers[index].input)*np.dot(delta_next, self.weights[index])
        return delta

    def shuffle(self):
        newindex = np.random.permutation(len(self.variables))
        self.teachers, self.variables = self.teachers[newindex], self.variables[newindex]
        """next_index=self.teachers.index.tolist()
        np.random.shuffle(next_index)
        self.variables.index,self.teachers.index=next_index,next_index
        self.variables,self.teachers=self.variables.sort_index(),self.teachers.sort_index()"""


class Error(Exception):
    def __init__(self, message):
        print(message)


class layer:

    """
    this class consider as j-th layer in below methods \n
    E mean 
    """

    def __init__(self, unit_size, func=lambda x: x, dfunc=lambda x: 1):
        """acts=[activater,dactivater]"""

        self.activater = func
        self.dactivater = dfunc
        # self.m_w,self.v_w,self.m_b,self.v_b=0,0,0,0
        self.input, self.output = 0, 0
        self.unit_size = unit_size
        self.input_check = False  # inputが更新されてoutputが非更新の時True

    def set_input(self, new_input):
        self.input = new_input
        if(self.input.shape[1] != self.unit_size):
            raise Error("mismatch columns number of input")

        self.input_check = True

    def generate_output(self):
        if(not self.input_check):
            raise Error("update input or turn input_check into True")
        self.output = self.activater(self.input)
        self.input_check = False
