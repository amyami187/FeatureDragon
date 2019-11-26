import pennylane as qml
import torch
import numpy as np

from torch.nn.parameter import Parameter
import torch.nn as nn

import builtins

import from circuit import circuit


class QCircuitNet(torch.nn.Module):
    def __init__(self):
        super(QCircuitNet, self).__init__()
        
        self.params=Parameter(torch.rand(2,dtype=torch.float32))
        
        


    def forward(self,X_args,Y_args):
        
        # evaluate on a sub batch
        for i_eval in range(args.size()[0]):
            exp_vals_list.append(circuit(self.params,X=X_args[i_eval], Y_args[i_eval]).float())
                                 
        exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return exp_vals



    


