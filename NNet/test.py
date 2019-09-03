import network
import activate_function as af
import improvement as im
import numpy as np

lay1 = network.layer(3, af.sig, af.dsig)
lay2 = network.layer(2, af.sig, af.dsig)
net = network.NeuralNetWork(lay1, lay2, imp=im.ADAM)

X = np.random.randn(10000, 3)
func=lambda x: x[0] + 2 * x[1] - x[2]
Y = np.apply_along_axis(func,1, X)

net.set_data(Y, X)
net.train(5, 3)

print(X[0:1])
print(net.forward_excute(X[0:1]))
print(net.layers[-1].output)