# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:39:54 2019

@author: Tomas G. Aquino
DRNN algorithm described in Haghi et al. (2019), NeurIPS
"""
###############################################################################
# Libraries and PyTorch setup
###############################################################################
import numpy as np

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.utils.data as data_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import RegressorMixin

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)

###############################################################################
# Recurrent neural network using Pytorch (Haghi et al., 2019 NeurIPS)
###############################################################################
class DRNN_Network(nn.Module):
    def __init__(self,n_features,batch_size,n_nodes,n_outputs,dropout_coef,mode):        
        # Passing arguments from parent class (nn.Module) automatically
        super(DRNN_Network, self).__init__()
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs        
        self.dropout_coef = dropout_coef
        self.mode = mode        
        self.Wx = nn.Linear(self.n_nodes,self.n_nodes,bias = False).cuda()
        self.Wr = nn.Linear(self.n_nodes,self.n_nodes,bias = False).cuda()
        self.Wu = nn.Linear(self.n_features,self.n_nodes,bias = False).cuda()
        self.Wz = nn.Linear(self.n_outputs, self.n_nodes,bias = False).cuda()
        self.tanh1 = nn.Tanh().cuda()
        self.dropout_layer = nn.Dropout2d(dropout_coef).cuda()
        self.bx = torch.nn.Parameter(torch.zeros(self.batch_size,
                                                 self.n_nodes)).cuda()
        self.Wo = nn.Linear(self.n_nodes,self.n_outputs).cuda()
        
    # Hidden layer setup
    # x: hidden state, r: non-linearly transformed state
    def init_hidden(self,):
        self.x = torch.FloatTensor(self.batch_size, self.n_nodes).type(
                torch.FloatTensor)
        # Filling x with random normal numbers
        init.normal_(self.x, 0.0, 0.01)
        self.x = Variable(self.x, requires_grad=True)        
        self.r = self.tanh1(self.x)
    
    # Recurrent network forward dynamics (one time step)      
    # u: data matrix (batch_size x time length x n_features)
    def forward(self, u, z, epsilon, f, enc):
        # Performing scheduled sampling (epsilon is current probability of
        # providing ground truth as feedback vs. the models predictions, 
        # for the current training epoch)
        if epsilon > 0:
            p = np.random.uniform(low=0, high=1, size=self.batch_size)
            p = np.where(p < epsilon, 0, 1)
            p = torch.from_numpy(p)
            p = p.to(device)
            # One-hot encode f to serve as input to network
            if self.mode == 'classification':                     
                one_hot_f = enc.fit_transform(f.cpu().reshape((f.shape[0],1)))
                input_f = torch.from_numpy(one_hot_f).to(device)       
                p = p.reshape((p.shape[0],1))
            elif self.mode == 'regression':
                input_f = f
            # Syntax: torch.where(condition, x if condition, else y)
            # z is the sampled output from the RNN (f: ground truth, z:
            # model estimate of f)
            z = torch.where(p == 0, input_f, z.double())                
            z = z.reshape((self.batch_size,self.n_outputs))
        # Updating network's hidden states
        self.x = self.x.cuda()
        self.r = self.r.cuda()
        u = u.cuda()
        z = z.cuda().float()
        self.x = self.Wx(self.x) + self.Wr(self.r) + self.Wu(u) + self.Wz(z) + self.bx
        self.r = self.tanh1(self.x)
        z = self.Wo(self.r)
        if self.mode == 'regression':
            # Taking non-linear output for abs values larger than 1
            z = torch.where(abs(z) > 1, self.tanh1(z), z)            
        return z
    
    
    
###############################################################################
# Decoder class with sklearn structure
###############################################################################

class drnn_decoder(RegressorMixin):
    def __init__(self,n_features,batch_size,n_nodes,n_outputs,dropout_coef,mode):
        # Model parameters
        self.n_features = n_features
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.dropout_coef = dropout_coef     
        self.mode = mode
        
        # Fitting parameters
        self.n_epochs = 10
        self.epoch_flip_threshold = 10
        self.epsilon_s = 0 #0.25
        self.epsilon_e = 0     
        self.regul_coef = 0 
        self.lr = 0.001          
        self.model = DRNN_Network(
                self.n_features,self.batch_size,self.n_nodes,self.n_outputs,self.dropout_coef,self.mode)
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),lr=self.lr,weight_decay=self.regul_coef)
        if self.mode == 'regression':        
            self.loss_function = torch.nn.MSELoss()
        elif self.mode == 'classification':
            self.loss_function = torch.nn.CrossEntropyLoss()
            self.enc = []
            # Adding one hot encoder for classification
            category_list = [[0,1]]
            self.enc = OneHotEncoder(handle_unknown='ignore',sparse=False,categories=category_list)

        
        
    # Fit recurrent neural network using backpropagation through time
    def fit(self,X,y):
        loss_by_epoch = []
        series_loss = np.zeros((X.shape[0],X.shape[1],self.n_epochs))
        zero_one_loss = np.zeros((X.shape[0],X.shape[1],self.n_epochs))
        # One-hot encoder is declared in case of classification        
        for epochI in range(self.n_epochs):
            print('epoch: ', epochI)
            # Defining scheduled sampling rate for this epoch
            # Only enforce epsilon decay up until a certain threshold
            if epochI <= self.epoch_flip_threshold:
                epsilon = ((self.epsilon_e - self.epsilon_s)/
                           self.epoch_flip_threshold)*epochI + self.epsilon_s
            else:
                epsilon = self.epsilon_e        
                
            model = self.model.to(device)
            model.train()            
            # Loading data onto PyTorch's structure
            u_train = Variable(torch.Tensor(X).type(torch.FloatTensor), requires_grad=False)
            f_train = Variable(torch.Tensor(y).type(torch.FloatTensor), requires_grad=False)
            dataset = data_utils.TensorDataset(u_train, f_train)         
            dataloader = data_utils.DataLoader(
                    dataset,batch_size=self.batch_size,shuffle=False,num_workers=0,drop_last=True)
            # Keeping track of stepwise loss
            running_loss = 0.0            
            batch_counter = 0                
            # PyTorch's dataloader automatically handles batching
            for i, data in enumerate(dataloader):                          
                print('dataI: ', i)
                self.optimizer.zero_grad()
                loss = 0
                u, f = data
                u, f = u.to(device), f.to(device)                
                z = torch.Tensor([0.5]).type(torch.FloatTensor)     
                # One hot encode labels for classification
                if self.mode == 'classification':                                         
                    target = f[:,0].long()
                    one_hot_z = self.enc.fit_transform(z.cpu().reshape((z.shape[0],1)))
                    input_z = torch.from_numpy(one_hot_z).to(device)                        
                elif self.mode == 'regression':                    
                    input_z = z
                    target = f[:,0]
                self.model.init_hidden() 
                # Loop model over time steps adding up loss
                n_timesteps = u.shape[1]
                # Adding dropout layer to network, ignoring if dropout is 0
                if model.dropout_coef > 0:
                    u = model.dropout_layer(u)                          
                z_series = input_z.repeat(n_timesteps,1)                                        
                for timeI in np.arange(n_timesteps):
                    input_f = f[:,timeI]
                    input_u = u[:,timeI,:]
                    input_z, = model.forward(input_u, input_z, epsilon, input_f, self.enc)
                    loss += self.loss_function(input_z.unsqueeze(0),target)
                    z_series[timeI] = input_z
                    
                loss.backward(retain_graph=True)
                self.optimizer.step()            
                running_loss += loss.detach().item()   
                batch_counter += 1
                # Getting loss for every timestep
                for timeI in np.arange(z_series.shape[0]):
                    _, predicted = torch.max(z_series[timeI].unsqueeze(0), 1)
                    zero_one_loss[i,timeI,epochI] = torch.abs(predicted-target)
                    series_loss[i,timeI,epochI] = self.loss_function(z_series[timeI].unsqueeze(0),target) 
            loss_by_epoch.append(running_loss)            
        self.loss_by_epoch = loss_by_epoch
        self.zero_one_loss = zero_one_loss
        self.series_loss = series_loss
                    
        
    def predict(self,X,y):
        self.model.eval()
        n_examples = X.shape[0]
        n_timesteps = X.shape[1]
        y_pred_hat = np.zeros((n_examples,n_timesteps))
        series_loss = np.zeros((X.shape[0],X.shape[1]))
        zero_one_loss = np.zeros((X.shape[0],X.shape[1]))
        # Loading data onto PyTorch's structure
        u_train = Variable(torch.Tensor(X).type(torch.FloatTensor), requires_grad=False)
        f_train = Variable(torch.Tensor(y).type(torch.FloatTensor), requires_grad=False)
        dataset = data_utils.TensorDataset(u_train, f_train)         
        dataloader = data_utils.DataLoader(
                dataset,batch_size=self.batch_size,shuffle=False,num_workers=0,drop_last=True)
        for i, data in enumerate(dataloader):                          
            print('dataI: ', i)                                
            u, f = data
            u, f = u.to(device), f.to(device)            
            z = torch.Tensor([0.5]).type(torch.FloatTensor)    
            # One hot encode labels for classification
            if self.mode == 'classification':                                         
                target = f[:,0].long()                
                one_hot_z = self.enc.fit_transform(z.cpu().reshape((z.shape[0],1)))
                input_z = torch.from_numpy(one_hot_z).to(device)                        
            elif self.mode == 'regression':                
                input_z = z
                target = f[:,0]
            self.model.init_hidden() 
            # Loop model over time steps adding up loss
            n_timesteps = u.shape[1]
            # Adding dropout layer to network, ignoring if dropout is 0
            if self.model.dropout_coef > 0:
                u = self.model.dropout_layer(u)                          
            z_series = input_z.repeat(n_timesteps,1)                                        
            for timeI in np.arange(n_timesteps):                    
                input_u = u[:,timeI,:]
                input_z, = self.model.forward(input_u, input_z, 0, [], self.enc)                
                z_series[timeI] = input_z
                
            # Getting loss for every timestep
            for timeI in np.arange(z_series.shape[0]):
                _, predicted = torch.max(z_series[timeI].unsqueeze(0), 1)
                zero_one_loss[i,timeI] = torch.abs(predicted-target)
                series_loss[i,timeI] = self.loss_function(z_series[timeI].unsqueeze(0),target)
                
            if self.mode == 'classification':
                _, predicted = torch.max(z_series, 1)
            else:
                predicted = z_series
            y_pred_hat[i,:] = predicted.cpu()
        return y_pred_hat
        
        
        '''


        
        self.model.eval()
        n_examples = X.shape[0]
        n_timesteps = X.shape[1]
        y_pred_hat = np.zeros((n_examples,n_timesteps))        
        # Loading features to use in prediction
        u_test = Variable(torch.Tensor(X).type(torch.FloatTensor), requires_grad=False)
        dataset = data_utils.TensorDataset(u_test)
        test_dataloader = data_utils.DataLoader(
                    dataset,batch_size=self.batch_size,shuffle=False,num_workers=0,drop_last=True)
        for i, data in enumerate(test_dataloader): 
            print('dataI: ', i) 
            u = data[0]
            # Initial state
            if i == 0:
                z = torch.Tensor([0.5]).type(torch.FloatTensor)
            if self.mode == 'classification':              
                if i == 0:                                                  
                    one_hot_z = self.enc.fit_transform(z.cpu().reshape((z.shape[0],1)))
                    input_z = torch.from_numpy(one_hot_z).to(device)                               
            elif self.mode == 'regression':
                if i == 0:
                    input_z = z                        
                                  
            self.model.init_hidden()
            z_series = input_z.repeat(n_timesteps,1)                                        
            # Time loop
            for timeI in np.arange(n_timesteps):
                input_u = u[:,timeI,:]                
                input_z, = self.model.forward(input_u, input_z, 0, [], self.enc)                    
                z_series[timeI] = input_z
        
            if self.mode == 'classification':
                _, predicted = torch.max(z_series, 1)
            else:
                predicted = z_series
            y_pred_hat[i,:] = predicted.cpu()
     
        return y_pred_hat
            
        
        '''
        