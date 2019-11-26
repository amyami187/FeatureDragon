import pennylane as qml
import torch
import numpy as np

from torch.nn.parameter import Parameter
import torch.nn as nn

import builtins


def H_phase(x):
    for i in range(x.size): 
        qml.Hadamard(wires=i)
        qml.RZ(x[i],wires=i)

def entangle_rotate(x):
    for i in range(x.size): 
        i_next=(i+1)%(x.size)
        #prevent reverse cycle for two qbit system
        if i==1 and i_next==0:
            #print('state prep: preventing trivial reverse cycle for a 2 qbit circuit')
            break

        phi=(np.pi-x[i])*(np.pi-x[i_next])
        #print('entangling {} and {} with phi rotation {}'.format(i,i_next,phi))
        qml.CNOT(wires=[i,i_next])
        qml.RZ(phi,wires=i_next)
        qml.CNOT(wires=[i,i_next])
        
def entangle_rotate_sq(x):
    for i in range(x.size): 
        i_next=(i+1)%(x.size)
        #prevent reverse cycle for two qbit system
        if i==1 and i_next==0:
            #print('state prep: preventing trivial reverse cycle for a 2 qbit circuit')
            break

        phi=np.sqrt((np.pi-x[i])*(np.pi-x[i_next]))
        #print('entangling {} and {} with phi rotation {}'.format(i,i_next,phi))
        qml.CNOT(wires=[i,i_next])
        qml.RZ(phi,wires=i_next)
        qml.CNOT(wires=[i,i_next])


def ryrz(params):
    n_wire=params.size//2
    for i_wire in range(n_wire):
        i_theta=i_wire*2
        i_phi=i_theta+1
        qml.RY(params[i_theta],wires=i_wire)
        qml.RZ(params[i_phi],wires=i_wire) 


def cycle_ent(n_wires):
    for i in range(n_wires): 
        i_next=(i+1)%(n_wires)
        if i==1 and i_next==0:
            #print('classifier: preventing trivial reverse cycle for a 2 qbit circuit')
            break
        #print('entangling {} and {}'.format(i,i_next))
        qml.CNOT(wires=[i,i_next])


def make_layers(params,depth,n_wires):
    #print('params size {} and xcheck {}'.format(params.size,((depth+1)*n_wires*2)))
    assert params.size==((depth+1)*n_wires*2)
    ryrz(params[:2*n_wires])
    for i_layer in range(depth):
        i_p_first=(i_layer+1)*2*n_wires
        i_p_last=(i_layer+2)*2*n_wires #slice indexing, so one past
        cycle_ent(n_wires)
        ryrz(params[i_p_first:i_p_last])


@qml.qnode(builtins.qpu,interface='torch')
def circuit(params,inputs=None,nlayers=None, nwires=None):
    #print(inputs)
    #qml.BasisState(np.asarray([0,0]),wires=[0,1])
    H_phase(inputs)
    entangle_rotate(inputs)
    make_layers(params,nlayers,nwires)
    return [qml.expval(qml.PauliZ(i)) for i in range(nwires)]


@qml.qnode(builtins.qpu,interface='torch')
def circuitBigU(params,inputs=None,nlayers=None, nwires=None):
    #print(inputs)
    #qml.BasisState(np.asarray([0,0]),wires=[0,1])
    #in the paper UH is applied twice
    H_phase(inputs)
    H_phase(inputs)
    entangle_rotate(inputs)
    make_layers(params,nlayers,nwires)
    return [qml.expval(qml.PauliZ(i)) for i in range(nwires)]


@qml.qnode(builtins.qpu,interface='torch')
def circuit_sq(params,inputs=None,nlayers=None, nwires=None):
    #print(inputs)
    #qml.BasisState(np.asarray([0,0]),wires=[0,1])
    H_phase(inputs)
    entangle_rotate_sq(inputs)
    make_layers(params,nlayers,nwires)
    return [qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1))]


class QCircuitNet(torch.nn.Module):
    def __init__(self,nwires,nlayers,nclasses):
        super(QCircuitNet, self).__init__()
        self.nwires=nwires
        self.nlayers=nlayers
        self.nclasses=nclasses
        self.n_q_params=((nlayers+1)*nwires*2)
        #self.qpu=qpu
        self.params=Parameter(torch.rand(self.n_q_params,dtype=torch.float32))
        self.fc1 = nn.Linear(nwires, nclasses)
        


    def forward(self,args):
        #call circuit on each example; build results tensor succesively
        #this seems very pedestrian but idk if there is a better way
        #print('args: {}'.format(args))
        exp_vals_list=[]
        for i_eval in range(args.size()[0]):
            exp_vals_list.append(circuit(self.params,inputs=args[i_eval],
                                         nlayers=self.nlayers,nwires=self.nwires).float())
                                 
        exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return self.fc1(exp_vals)
    
class QCircuitNetBigU(torch.nn.Module):
    def __init__(self,nwires,nlayers,nclasses):
        super(QCircuitNetBigU, self).__init__()
        self.nwires=nwires
        self.nlayers=nlayers
        self.nclasses=nclasses
        self.n_q_params=((nlayers+1)*nwires*2)
        #self.qpu=qpu
        self.params=Parameter(torch.rand(self.n_q_params,dtype=torch.float32))
        self.fc1 = nn.Linear(nwires, nclasses)
        


    def forward(self,args):
        #call circuit on each example; build results tensor succesively
        #this seems very pedestrian but idk if there is a better way
        #print('args: {}'.format(args))
        exp_vals_list=[]
        for i_eval in range(args.size()[0]):
            exp_vals_list.append(circuitBigU(self.params,inputs=args[i_eval],
                                             nlayers=self.nlayers,nwires=self.nwires).float())
                                 
        exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return self.fc1(exp_vals)
    
    
class QCircuitNetUp(torch.nn.Module):
    def __init__(self,nfeatures,nwires,nlayers,nclasses):
        super(QCircuitNetUp, self).__init__()
        self.nfeatures=nfeatures
        self.nwires=nwires
        self.nlayers=nlayers
        self.nclasses=nclasses
        self.n_q_params=((nlayers+1)*nwires*2)
        #self.qpu=qpu
        self.sigmoid = nn.Sigmoid()
        self.params=Parameter(torch.rand(self.n_q_params,dtype=torch.float32))
        self.fc0 = nn.Linear(nfeatures,nwires)
        self.fc1 = nn.Linear(nwires, nclasses)
        


    def forward(self,args):
        #up/downscale to desired number of qbits
        #use sigmoid to scale 0-pi
        args=self.fc0(args)
        args=np.pi*self.sigmoid(args)
        exp_vals_list=[]
        #call circuit on each example; build results tensor succesively
        #this seems very pedestrian but idk if there is a better way
        #print('args: {}'.format(args))
        for i_eval in range(args.size()[0]):
            exp_vals_list.append(circuit(self.params,inputs=args[i_eval],
                                         nlayers=self.nlayers,nwires=self.nwires).float())
                                 
        exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return self.fc1(exp_vals)


class QCircuitNet_sq(torch.nn.Module):
    def __init__(self,nwires,nlayers,nclasses):
        super(QCircuitNet_sq, self).__init__()
        self.nwires=nwires
        self.nlayers=nlayers
        self.nclasses=nclasses
        self.n_q_params=((nlayers+1)*nwires*2)
        #self.qpu=qpu
        self.params=Parameter(torch.rand(self.n_q_params,dtype=torch.float32))
        self.fc1 = nn.Linear(nwires, nclasses)
        


    def forward(self,args):
        #call circuit on each example; build results tensor succesively
        #this seems very pedestrian but idk if there is a better way
        #print('args: {}'.format(args))
        exp_vals_list=[]
        for i_eval in range(args.size()[0]):
            exp_vals_list.append(circuit_sq(self.params,inputs=args[i_eval],
                                            nlayers=self.nlayers,nwires=self.nwires).float())
                                 
        exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return self.fc1(exp_vals)



    


