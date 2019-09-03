import numpy as np


class improve_method:
    def __init__(self):
        pass

    def improve(self, weight, E):
        pass

    def clear(self):
        self.__init__()


class ADAM(improve_method):
    epsilon, alpha = 10**(-8), 0.001
    beta_1, beta_2 = 0.9, 0.999

    def __init__(self):
        self.m, self.v, self.time = None, None, 0

    def update(self, weight, E):
        self.time += 1

        if(self.time == 1):
            self.m, self.v = np.zeros(E.shape), np.zeros(E.shape)

        for i in range(E.shape[0]):
            self.m[i] = self.beta_1*self.m[i]+(1-self.beta_1)*E[i]
            self.v[i] = self.beta_2*self.v[i]+(1-self.beta_2)*(E[i]**2)

        m_, v_ = self.m/(1-self.beta_1**self.time), self.v / \
            (1-self.beta_2**self.time)
        weight -= self.alpha*m_/(np.sqrt(v_)+self.epsilon)

        return weight


class GSD(improve_method):
    def __init__(self, rate):
        self.rate = rate

    def update(self, weight, E):
        weight -= self.rate*E
        return weight
