
# coding: utf-8

# In[ ]:


#from sympy.physics.quantum import TensorProduct as tensor
from pennylane import numpy as np
import pennylane as qml
from modules.featuremap4datapoints import featuremap
from sklearn.preprocessing import normalize
import sys
import time
from modules.circ_fns import *

import torch

import builtins
U1f = U1f()
U1b = U1b()
U2f = U2f()
U2b = U2b()
U3f = U3f()
U3b = U3b()
U4f = U4f()
U4b = U4b()
U5f = U5f()
U5b = U5b()
U6f = U6f()
U6b = U6b()
U7f = U7f()
U7b = U7b()
U8f = U8f()
U8b = U8b()

all_wires=list(range(8))

@qml.qnode(builtins.dev_qubit,interface='torch')
def circuit(phi0, Xdata=None, Y=None):
    X1 = Xdata[0:, 0]
    X2 = Xdata[0:, 1]
    for i in range(4):
        qml.Hadamard(wires=i)

    qml.QubitUnitary(U1f, wires=all_wires)
    featuremap(X1[0], X2[0], Y[0], phi0)
    qml.QubitUnitary(U1b, wires=all_wires)

    qml.QubitUnitary(U2f, wires=all_wires)
    featuremap(X1[1], X2[1], Y[1], phi0)
    qml.QubitUnitary(U2b, wires=all_wires)

    qml.QubitUnitary(U3f, wires=all_wires)
    featuremap(X1[2], X2[2], Y[2], phi0)
    qml.QubitUnitary(U3b, wires=all_wires)

    qml.QubitUnitary(U4f, wires=all_wires)
    featuremap(X1[3], X2[3], Y[3], phi0)
    qml.QubitUnitary(U4b, wires=all_wires)

    qml.QubitUnitary(U5f, wires=all_wires)
    featuremap(X1[4], X2[4], Y[4], phi0)
    qml.QubitUnitary(U5b, wires=all_wires)

    qml.QubitUnitary(U6f, wires=all_wires)
    featuremap(X1[5], X2[5], Y[5], phi0)
    qml.QubitUnitary(U6b, wires=all_wires)

    qml.QubitUnitary(U7f, wires=all_wires)
    featuremap(X1[6], X2[6], Y[6], phi0)
    qml.QubitUnitary(U7b, wires=all_wires)

    qml.QubitUnitary(U8f, wires=all_wires)
    featuremap(X1[7], X2[7], Y[7], phi0)
    qml.QubitUnitary(U8b, wires=all_wires)

    qml.Hadamard(wires=0)

    return qml.expval(qml.Hermitian(np.array([[1, 0], [0, 0]]), wires=0)), qml.expval(qml.PauliZ(wires=5))

