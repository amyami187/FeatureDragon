import pennylane as qml
import torch
import numpy as np

from torch.nn.parameter import Parameter
import torch.nn as nn

import builtins

from modules.circ_4datapoints import circuit


class QCircuitNet_4eg(torch.nn.Module):
    def __init__(self):
        super(QCircuitNet_4eg, self).__init__()
        
        self.params=Parameter(torch.rand(12,dtype=torch.float32))
        print(self.params)
        
        


    def forward(self,X_args,Y_args):
        
        #print('data inside torch wrapper: X_args {}, Y_args {}'.format(X_args,Y_args))
        # evaluate on a sub batch
        #for i_eval in range(args.size()[0]):
        #exp_vals.append(circuit(self.params,X=X_args[i_eval], Y=Y_args[i_eval]).float())
        #exp_vals=circuit(self.params,Xdata=X_args, Y=Y_args).float()
        exp_vals=circuit(self.params,Xdata=X_args, Y=Y_args)
        exp_vals_float=exp_vals.float()
        #print('circuit output values: {} {}'.format(exp_vals_float[0],exp_vals_float[1]))
        out=exp_vals_float[0]*exp_vals_float[1]
        #print('model out: {}'.format(out))
                                 
        #exp_vals=torch.stack(exp_vals_list,dim=0)
        #print('exp_vals: {}'.format(exp_vals))
        return out



    


