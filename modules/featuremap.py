
# coding: utf-8

# In[ ]:

import pennylane as qml

"""def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[4, 5])  # flip label qubit
    qml.CRot(phi0.item(0), phi0.item(1), phi0.item(2), wires=[6, 3])
    qml.CRot(x1, x2, 0, wires=[4, 3])
    qml.CRot(phi0.item(3), phi0.item(4), phi0.item(5), wires=[6, 4])
    qml.CRot(x1, x2, 0, wires=[6, 4])
    qml.CZ(wires=[3, 4])
    qml.CZ(wires=[4, 3])
    # layer 2
    qml.CRot(phi0.item(6), phi0.item(7), phi0.item(8), wires=[6, 3])
    qml.CRot(x1, x2, 0, wires=[6, 3])
    qml.CRot(phi0.item(9), phi0.item(10), phi0.item(11), wires=[6, 4])
    qml.CRot(x1, x2, 0, wires=[6, 4])
    qml.CZ(wires=[3, 4])
    qml.CZ(wires=[4, 3])
"""
def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[4, 5])  # flip label qubit
    qml.CRX(x1, wires = [4,2])    
    qml.CRX(x2, wires = [4,3])
    qml.CRX(phi0[0], wires = [4,2])
    qml.CRX(phi0[1], wires = [4,3])
    qml.CRX(x1, wires = [4,2])    
    qml.CRX(x2, wires = [4,3])
    qml.CRX(phi0[2], wires = [4,2])
    qml.CRX(phi0[3], wires = [4,3])

