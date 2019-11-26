# %%

# coding: utf-8

# %%


#from sympy.physics.quantum import TensorProduct as tensor
from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize
import torch

import builtins
from modules.featuremap import featuremap


# dev_qubit = qml.device('default.qubit', wires=6)
# Hadamard classifier circuit for 4 data points with 1 new input to be classified and any arbitray featuremap
@qml.qnode(builtins.dev_qubit, interface='torch')
def circuit(phi0, Xdata=None, Y=None):
    X1 = Xdata[0:, 0]
    X2 = Xdata[0:, 1]
    #print("data inside circuit: X1: {} X2: {}".format(X1,X2))
    for i in range(2):
        qml.Hadamard(wires=i)

    qml.PauliX(wires=[0])
    qml.PauliX(wires=[1])
    qml.Toffoli(wires=[0,1,5])
    featuremap(X1[0], X2[0], Y[0], phi0)
    qml.Toffoli(wires=[0,1,5])
    qml.PauliX(wires=[0])
    qml.PauliX(wires=[1])

    qml.PauliX(wires = 0)
    qml.Toffoli(wires=[0,1,5])
    featuremap(X1[1], X2[1], Y[1], phi0)
    qml.Toffoli(wires=[0,1,5])
    qml.PauliX(wires = 0)

    qml.PauliX(wires = 1)
    qml.Toffoli(wires=[0,1,5])
    featuremap(X1[2], X2[2], Y[2], phi0)
    qml.Toffoli(wires=[0,1,5])
    qml.PauliX(wires = 1)

    qml.Toffoli(wires=[0,1,5])
    featuremap(X1[3], X2[3], Y[3], phi0)
    qml.Toffoli(wires=[0,1,5])

    qml.Hadamard(wires=0)

    return qml.expval(qml.Hermitian(np.array([[1, 0], [0, 0]]), wires=0)), qml.expval(qml.PauliZ(wires=4))

