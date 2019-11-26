'''
Author: Wojciech Fedorko
Stolen from multiple previous projects
'''

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import os
import sys
import time
import numpy as np

# from utils.data_handling import WCH5Dataset

from modules.logging_utils import CSVData





class Engine:
    """The training engine 
    
    Performs training and evaluation
    """

    def __init__(self, model, dataset, config):
        self.model = model
        if (config.device == 'gpu'):
            print("Requesting a GPU")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("CUDA is available")
            else:
                self.device=torch.device("cpu")
                print("CUDA is not available")
        else:
            print("Sticking to CPU")
            self.device=torch.device("cpu")

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion=nn.MSELoss()
        self.softmax = nn.Softmax(dim=1)

        #placeholders for data and labels
        self.data=None
        self.labels=None
        self.iteration=None


        self.dset=dataset

        self.train_dldr=DataLoader(self.dset,
                                   batch_size=config.batch_size_train+1,
                                   shuffle=False,
                                   sampler=SubsetRandomSampler(self.dset.train_indices),
                                   num_workers=config.num_workers_train,
                                   drop_last=True)
        
        self.train_eval_dldr=DataLoader(self.dset,
                                       batch_size=config.batch_size_train,
                                       shuffle=False,
                                       sampler=SubsetRandomSampler(self.dset.train_indices),
                                       num_workers=config.num_workers_train,
                                       drop_last=True)
        self.train_eval_iter=iter(self.train_eval_dldr)
        
        
        self.val_dldr=DataLoader(self.dset,
                                 batch_size=config.batch_size_val,
                                 shuffle=False,
                                 sampler=SubsetRandomSampler(self.dset.val_indices),
                                 num_workers=config.num_workers_val,
                                 drop_last=True)
        self.val_iter=iter(self.val_dldr)
        
        self.test_dldr=DataLoader(self.dset,
                                  batch_size=config.batch_size_test,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(self.dset.test_indices),
                                  num_workers=config.num_workers_test,
                                  drop_last=True)
        self.test_iter=iter(self.val_dldr)

        self.dirpath=config.dump_path + "/"+time.strftime("%Y%m%d_%H%M%S") + "/"

                
        try:
            os.stat(self.dirpath)
        except:
            print("Creating a directory for run dump: {}".format(self.dirpath))
            os.makedirs(self.dirpath,exist_ok=True)

        self.config=config
        
        # Save a copy of the config in the dump path
        f_config=open(self.dirpath+"/config_log.txt","w")
        f_config.write(str(vars(config)))


    def forward(self,train=True):
        """
        Args: self should have attributes, model, criterion, softmax, data, label
        Returns: a dictionary of predicted labels, softmax, loss, and accuracy
        """
        with torch.set_grad_enabled(train):
            # Move the data and the labels to the GPU
            # if using CPU this has no effect
            self.data = self.data.to(self.device)
            self.label = self.label.to(self.device)
            
            #this manipulates the data in the way the quantum circuit wants it
            data_adj=torch.zeros(self.data.shape[0]+1,self.data.shape[1])
            data_adj[:-1]=self.data[:]
            data_adj[-1]=self.data[-1]
            
            labels_adj=torch.zeros(self.label.shape[0]+1)
            labels_adj[:-1]=self.label[:]
            labels_adj[-1]=self.label[-1]
            

            
            model_out = self.model(data_adj,labels_adj)
            # Training
            
            self.loss = self.criterion(model_out,self.label[-1])
            
            prediction = torch.where(model_out>0.0,torch.tensor([1.0]),torch.tensor([-1.0]))
            #accuracy   = (prediction == self.label).sum().item() / float(prediction.nelement())        
            prediction = prediction.cpu().numpy()
        
        return {'prediction' : prediction,
                'loss'       : self.loss.detach().cpu().item()}

    def backward(self):
        self.optimizer.zero_grad()  # Reset gradients accumulation
        self.loss.backward()
        self.optimizer.step()
        
    # ========================================================================
    def train(self, epochs=3.0, report_interval=10, valid_interval=1000):
        # Based on WaTCHMaL workshop and W's code
        
        # Keep track of the validation accuracy
        best_val_acc = 0.0
        best_val_loss=1.0e6
        
        
        # Prepare attributes for data logging
        self.train_log, self.val_log = CSVData(self.dirpath+"log_train.csv"), CSVData(self.dirpath+"log_val.csv")
        # Set neural net to training mode
        self.model.train()
        # Initialize epoch counter
        epoch = 0.
        # Initialize iteration counter
        self.iteration = 0
        # Training loop
        while ((int(epoch+0.5) < epochs) ):
            print('Epoch',int(epoch+0.5),'Starting @',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            j = 0
            # Loop over data samples and into the network forward function
            for i, data in enumerate(self.train_dldr):

                # once in a while run valiation
                # as a sanity check run validation before we start training
                if i%valid_interval == 0:
                    self.model.eval()
                    
                    tot_ens_loss=0.0
                    for i_eval in range(self.config.num_val_samples):
                        try:
                            val_data_label = next(self.val_iter)
                        except StopIteration:
                            print("starting over on the validation set")
                            self.val_iter=iter(self.val_dldr)
                            val_data_label = next(self.val_iter)
                        
                        val_data=val_data_label[0]
                        val_label=val_data_label[1]
                        
                        
                        for i_train_ens in range(self.config.num_train_ensembles):
                            try:
                                ens_train_data = next(self.train_eval_iter)
                            except StopIteration:
                                print("starting over on the training set")
                                self.train_eval_iter=iter(self.train_eval_dldr)
                                ens_train_data = next(self.train_eval_iter)
                                
                            ens_data=ens_train_data[0]
                            ens_label=ens_train_data[1]
                            
                        
                            # Data and label
                            self.data = torch.cat((ens_data,val_data))
                            self.label = torch.cat((ens_label,val_label))
                    
                            res = self.forward(False)
                            tot_ens_loss+=res['loss']
                    avg_val_loss=tot_ens_loss/(self.config.num_val_samples*self.config.num_train_ensembles)
                    print('... Iteration %d ... Epoch %1.2f ... Validation Loss %1.3f' % (self.iteration,epoch,avg_val_loss))
                    
                    
                    self.model.train()

                    self.save_state()
                    mark_best=0
                    if avg_val_loss<best_val_loss:
                        best_val_loss=avg_val_loss
                        print('best validation loss so far!: {}'.format(best_val_loss))
                        self.save_state(best=True)
                        mark_best=1

                    self.val_log.record(['iteration','epoch','loss','saved_best'],[self.iteration,epoch,avg_val_loss,mark_best])
                    self.val_log.write()
                    self.val_log.flush()
                
                # Data and label
                self.data = data[0]
                self.label = data[1]
                
                
                # Call forward: make a prediction & measure the average error
                res = self.forward(True)
                # Call backward: backpropagate error and update weights
                self.backward()
                # Epoch update
                epoch += 1./len(self.train_dldr)
                self.iteration += 1
                
                # Log/Report
                #
                # Record the current performance on train set
                self.train_log.record(['iteration','epoch','loss'],[self.iteration,epoch,res['loss']])
                self.train_log.write()
                self.train_log.flush()
                
                # once in a while, report
                if i==0 or i%report_interval == 0:
                    print('... Iteration %d ... Epoch %1.2f ... Loss %1.3f' % (self.iteration,epoch,res['loss']))
                    pass
                    
                
                        
                if epoch >= epochs:
                    break
                                 
                    
        self.val_log.close()
        self.train_log.close()
        #np.save(self.dirpath + "/optim_state_array.npy", np.array(optim_state_list))
    
    # ========================================================================

    # Function to test the model performance on the validation
    # dataset ( returns loss, acc, confusion matrix )
    def validate(self, save_data=False):
        """
        Test the trained model on the validation set.
        
        Parameters: None
        
        Outputs : 
            total_val_loss = accumulated validation loss
            avg_val_loss = average validation loss
            total_val_acc = accumulated validation accuracy
            avg_val_acc = accumulated validation accuracy
            
        Returns : None
        """
       
        
        # Variables to output at the end
        val_loss = 0.0
        val_acc = 0.0
        val_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, accuracy, labels, predictions, softmaxes, examples= [],[],[],[],[],[]
            
            # Extract the event data and label from the DataLoader iterator
            for it, val_data in enumerate(self.val_dldr):
                
                sys.stdout.write("val_iterations : " + str(val_iterations) + "\n")
                
                self.data, self.label = val_data[0:2]
                

                # Run the forward procedure and output the result
                result = self.forward(False)
                val_loss += result['loss']
                val_acc += result['accuracy']
                
                # Add item to priority queues if necessary
                
                # Copy the tensors back to the CPU
                self.label = self.label.to("cpu")
                
                # Add the local result to the final result
                labels.extend(self.label)
                if save_data:
                    examples.extend(self.data.to("cpu").numpy())
                predictions.extend(result['prediction'])
                softmaxes.extend(result["softmax"])
                
                val_iterations += 1
                
        print(val_iterations)

        print("\nTotal val loss : ", val_loss,
              "\nTotal val acc : ", val_acc,
              "\nAvg val loss : ", val_loss/val_iterations,
              "\nAvg val acc : ", val_acc/val_iterations)
        
        np.save(self.dirpath + "labels.npy", np.array(labels))
        if save_data:
            np.save(self.dirpath + "examples.npy",np.array(examples))
        np.save(self.dirpath + "predictions.npy", np.array(predictions))
        np.save(self.dirpath + "softmax.npy", np.array(softmaxes))  
        
        
    # ========================================================================
    
            
    def save_state(self,best=False):
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     ("BEST" if best else ""),
                                     ".pth")
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict()
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_state(self, weight_file):
        
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)
            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            # load network weights
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load iteration count
            self.iteration = checkpoint['global_step']
        print('Restoration complete.')

