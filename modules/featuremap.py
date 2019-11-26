
# coding: utf-8

# In[ ]:

import pennylane as qml

def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[5, 4])  # flip label qubit
    qml.CRX(x1, wires = [4,2])    
    qml.CRX(x2, wires = [4,3])
    qml.CRX(phi0[0], wires = [4,2])
    qml.CRX(phi0[1], wires = [4,3])

