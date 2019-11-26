
# coding: utf-8

# In[ ]:


from sympy.physics.quantum import TensorProduct as tensor
from pennylane import numpy as np
import pennylane as qml
#from _featuremap import featuremap
from sklearn.preprocessing import normalize
import sys
import time
from . import circ_fns

@qml.qnode(dev_qubit)
def circuit(phi0, Xdata=None, Y=None):
    X1 = Xdata[0:, 0]
    X2 = Xdata[0:, 1]
    for i in range(4):
        qml.Hadamard(wires=i)

    qml.QubitUnitary(U1f(), wires=range(8))
    featuremap(X1[0], X2[0], Y[0], phi0)
    qml.QubitUnitary(U1b(), wires=range(8))

    qml.QubitUnitary(U2f(), wires=range(8))
    featuremap(X1[1], X2[1], Y[1], phi0)
    qml.QubitUnitary(U2b(), wires=range(8))

    qml.QubitUnitary(U3f(), wires=range(8))
    featuremap(X1[2], X2[2], Y[2], phi0)
    qml.QubitUnitary(U3b(), wires=range(8))

    qml.QubitUnitary(U4f(), wires=range(8))
    featuremap(X1[3], X2[3], Y[3], phi0)
    qml.QubitUnitary(U4b(), wires=range(8))

    qml.QubitUnitary(U5f(), wires=range(8))
    featuremap(X1[4], X2[4], Y[4], phi0)
    qml.QubitUnitary(U5b(), wires=range(8))

    qml.QubitUnitary(U6f(), wires=range(8))
    featuremap(X1[5], X2[5], Y[5], phi0)
    qml.QubitUnitary(U6b(), wires=range(8))

    qml.QubitUnitary(U7f(), wires=range(8))
    featuremap(X1[6], X2[6], Y[6], phi0)
    qml.QubitUnitary(U7b(), wires=range(8))

    qml.QubitUnitary(U8f(), wires=range(8))
    featuremap(X1[7], X2[7], Y[7], phi0)
    qml.QubitUnitary(U8b(), wires=range(8))

    qml.Hadamard(wires=0)

    return qml.expval(qml.Hermitian(np.array([[1, 0], [0, 0]]), wires=0)), qml.expval(qml.PauliZ(wires=5))

