
# coding: utf-8

# In[ ]:


def featuremap(x1, x2, y, phi0):
    # encode y label
    if y == 1:
        qml.CNOT(wires=[5, 4])  # flip label qubit
    qml.RX(phi0[0], wires = [2])
    qml.RX(phi0[0], wires = [3])

