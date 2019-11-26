
# coding: utf-8

# In[ ]:


from sympy.physics.quantum import TensorProduct as tensor
from pennylane import numpy as np
import pennylane as qml
from sklearn.preprocessing import normalize
import sys
import time

I = np.array([[1.,0.], [0., 1.]])
zero = np.array([[1., 0.], [0., 0.]])
one = np.array([[0., 0.],[0., 1.]])
X = np.array([[0., 1.], [1., 0.]])

I_2 = np.eye(4)
I_3 = np.eye(8)
I_4 = np.eye(16)
I_5 = np.eye(32)
I_6 = np.eye(64)


# T1 conditional on 1,1 - acts on qubits 1,2 and flips 8
U_T11 = tensor(zero, tensor(zero, I_6)) +         tensor(zero, tensor(one, I_6)) +         tensor(one, tensor(zero, I_6)) +         tensor(one, tensor(one, tensor(I_5, X)))


# T2 conditional on 1,0 - acts on qubits 1,2 and flips 8
U_T21 = tensor(zero,tensor(zero,I_6)) +         tensor(zero,tensor(one, I_6)) +         tensor(one,tensor(zero,tensor(I_5, X))) +         tensor(one,tensor(one, I_6))
# T3 conditional on 0,1 - acts on qubits 1,2 and flips 8
U_T31 = tensor(zero,tensor(zero, I_6)) +         tensor(zero,tensor(one,tensor(I_5, X))) +         tensor(one,tensor(zero, I_6)) +         tensor(one,tensor(one,tensor(I_5,X)))
# T4 conditional on 0,0 - acts on qubits 1,2 and flips 8
U_T41 = tensor(zero,tensor(zero,tensor(I_5, X))) +         tensor(zero,tensor(one, I_6)) +         tensor(one,tensor(zero, I_6)) +         tensor(one,tensor(one, I_6))

# T1 conditional on 1,1 - acts on qubits 3,8 and flips 7
U_T12 = tensor(I_2, tensor(zero, tensor(I_4, zero))) +         tensor(I_2, tensor(zero, tensor(I_4, one))) +         tensor(I_2, tensor(one, tensor(I_4, zero))) +         tensor(I_2, tensor(one, tensor(I_3, tensor(X, one))))
# T3 conditional on 0,1 - acts on qubits 3,8 and flips 7
U_T32 = tensor(I_2, tensor(zero, tensor(I_4, zero))) +         tensor(I_2, tensor(zero, tensor(I_3, tensor(X, one)))) +         tensor(I_2, tensor(one, tensor(I_4, zero))) +         tensor(I_2, tensor(one, tensor(I_4, one)))


# Sequences of toffolis indexed and f = forward, b = backward: #####

def U1f():
    return np.matmul(U_T32, U_T41)
    
def U1b(): 
    return np.linalg.inv(U1f)

def U2f(): 
    return np.matmul(U_T12, U_T41)

def U2b(): 
    return np.linalg.inv(U2f)

def U3f(): 
    return np.matmul(U_T32, U_T31)

def U3b(): 
    return np.linalg.inv(U3f)

def U4f(): 
    return np.matmul(U_T12, U_T31)

def U4b(): 
    return np.linalg.inv(U4f)

def U5f(): 
    return np.matmul(U_T32, U_T21)

def U5b(): 
    return np.linalg.inv(U5f)

def U6f(): 
    return np.matmul(U_T12, U_T21)

def U6b(): 
    return np.linalg.inv(U6f)

def U7f(): 
    return np.matmul(U_T32, U_T11)

def U7b(): 
    return np.linalg.inv(U7f)

def U8f(): 
    return np.matmul(U_T12, U_T11)

def U8b(): 
    return np.linalg.inv(U8f)

