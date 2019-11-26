# %%

# coding: utf-8

# %%

import pennylane as qml

def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[5, 4])  # flip label qubit
    qml.CRot(phi0(0), phi0(1), phi0(2), wires=[4, 2])
    qml.CRot(x1, x2, 0, wires=[4, 2])
    qml.CRot(phi0(3), phi0(4), phi0(5), wires=[4, 3])
    qml.CRot(x1, x2, 0, wires=[4, 3])
    qml.CZ(wires=[2, 3])
    qml.CZ(wires=[3, 2])
    # layer 2
    qml.CRot(phi0(6), phi0(7), phi0(8), wires=[4, 2])
    qml.CRot(x1, x2, 0, wires=[4, 2])
    qml.CRot(phi0(9), phi0(10), phi0(11), wires=[4, 3])
    qml.CRot(x1, x2, 0, wires=[4, 3])
    qml.CZ(wires=[2, 3])
    qml.CZ(wires=[3, 2])
"""
def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[5, 4])  # flip label qubit
    qml.CRX(x1, wires = [4,2])    
    qml.CRX(x2, wires = [4,3])
    qml.CRX(phi0[0], wires = [4,2])
    qml.CRX(phi0[1], wires = [4,3])
    qml.CRX(x1, wires = [4,2])    
    qml.CRX(x2, wires = [4,3])
    qml.CRX(phi0[2], wires = [4,2])
    qml.CRX(phi0[3], wires = [4,3])
"""
