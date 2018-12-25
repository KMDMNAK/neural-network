import numpy as np

def sig(x):
    return 1/(1+np.exp(-x))
def dsig(x):
    s=sig(x)
    return (1-s)*s

def swish(x):
    return (x*sig(x))
def dswish(x):
    s=sig(x)
    return (1+(1-s)*x)*s

def tanh(x):
    return np.tanh(x)
def dtanh(x):
    return 1/(np.cosh(x)**2)

def ReLU(x):
    x[np.where(x<0)]=0
    return x
def dReLU(x):
    x[np.where(x<0)]=0
    x[np.where(x>0)]=1
    return x

def softmax(x):
    b=1/np.exp(-x)
    print((b.T/b.sum(axis=1)).T)
    