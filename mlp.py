#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 11:09:15 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


class Net(nn.Module):
    
    def __init__(self, input_dim, layer_wid, nonlinearity):
        """
        
        :param input_dim: Input dimension
        :param layer_wid: List of numbers that represent number of neurons in
            each layer, last item in the list is output dimension
        """
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = layer_wid[-1]
        
        self.fc_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(in_features=input_dim, 
                                        out_features=layer_wid[0]))
        
        for i in range(len(layer_wid) - 1):
            self.fc_layers.append(nn.Linear(in_features=layer_wid[i], 
                                            out_features=layer_wid[i + 1]))

        # for fc in self.fc_layers:
        #     nn.init.constant_(fc.weight,1)

        # Find total number of parameters
        num_param=0
        num_parami = []
        for p in self.parameters():
            temp=1
            for s in p.size():
                temp *= s
            num_parami.append(temp)    
            num_param += temp
            
        self.num_param = num_param
        
        # Set the nonlinearity
        if nonlinearity == "sigmoid":
            self.nonlinearity = lambda x: torch.sigmoid(x)
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: F.relu(x)
        elif nonlinearity == "softplus":
            self.nonlinearity = lambda x: F.softplus(x)
        elif nonlinearity == 'tanh':
            self.nonlinearity = lambda x: F.tanh(x)
        elif nonlinearity == 'leaky_relu':
            self.nonlinearity = lambda x: F.leaky_relu(x)

        self.criterion = nn.MSELoss()

    def test(self, X, y):
        """
        Find the RMSE for y
        """
        return F.mse_loss(self(X),y)**.5
    
    def get_param(self):
        """
        Return weights of this net as a vector
        """
        param_vector = torch.tensor([])  # Initial weights
        for param in self.parameters():
            param_vector = torch.cat((param_vector, param.view(-1)))
    
        return param_vector
    
    def set_param(self, param_vec):
        """
        Given a vector of parameters, sets the parameters of self
        """
        i = 0
        for param in self.parameters():
            j = param.nelement()
            param.data = param_vec[i:i + j].view(param.size())
            i += j
            
    def forward(self, x):
        """
        :param x: input with dimensions N x input_dim where N is number of
            inputs in the batch.
        """
        
        for fc_layer in self.fc_layers[:-1]:
            x = self.nonlinearity(fc_layer(x))
            
        return self.fc_layers[-1](x)


# class Network(Net):
#     def __init__(self,input_dim, layer_wid, nonlinearity, params):
#         super().__init__(input_dim, layer_wid, nonlinearity)
#         self.params = params
#         self.type=params['type']

#         if params['type']=='MLP':
#             if params['optimizer'] == "SGD":
#                 self.optimize = optim.SGD(self.parameters(), lr=params['lr'])
#             elif params['optimizer'] == "Adam":
#                 self.optimize = optim.Adam(self.parameters(), lr=params['lr'], betas=(0.9, 0.99),weight_decay=params['weight_decay'])
#             self.step = self.step_mlp
#         elif params['type']=='EKF':
#             if type(params['P'])==str:
#                 if params['P'] == "1":
#                     self.P_minus = torch.eye(self.num_param)  # independent Error covariance on state vec
#                 elif params['P'] == "yes_cor":            
#                     self.P_minus = torch.rand(self.num_param,self.num_param) # not independent
#                 elif params['P'] == "half":
#                     self.P_minus = torch.eye(self.num_param)*.5 # half diagnoal 
#                 elif params['P'] == "0.1":
#                     self.P_minus = torch.eye(self.num_param)*.1
#                 elif params['P'] == "0.01":
#                     self.P_minus = torch.eye(self.num_param)*0.01
#             else:
#                 self.P_minus = params['P']  

#             self.state_minus = self.get_param()
#             Q = torch.eye(self.state_minus.nelement()) * params['q']
#             self.Q =  Q
#             R = torch.eye(layer_wid[-1]) * params['r']**2
#             self.R = R
#             self.step = self.step_ekf

#     def step_mlp(self,X,y):
#         """
#         Update weights once with given samples
#         :param X: Nxp samples (N can be 1)
#         :param y: Corresponding labels (for regression Nx1)
#         """
        
#         y_prime = self(X)
#         loss = self.criterion(y_prime, y)
        
#         self.optimize.zero_grad()
#         self.zero_grad()
#         # self.mlp.zero_grad()
        
#         loss.backward(retain_graph=1)
#         self.optimize.step()

#     def step_ekf(self, X, y):
#         """
#         Update weights once with given samples
#         :param X: Nxp samples (N can be 1)
#         :param y: Corresponding labels (for regression Nx1)
#         :param Q: Noise covariance in state transition
#         :param R: Noise covariance in measurement equation
#         """
#         out = self(X)
#         # Construct the jacobian in measurement equation
#         H = []
#         for dim in out:  # For each output dimension, find the partial derivatives
#             h = torch.tensor([])  # h is a row in the jacobian H
#             self.zero_grad()
#             dim.backward(retain_graph=1)
#             for param in self.parameters():
#                 h = torch.cat((h, param.grad.view(-1)))
#             H.append(h)
#         H = torch.stack(H)

#         # Time update
#         self.state_predict = self.state_minus
#         self.P_predict = self.P_minus + self.Q

#         # Measurement update
#         nu = y - out  # innovations
#         psi = torch.mm(torch.mm(H, self.P_predict), H.t()) + self.R  # Innovations covariance
#         K = torch.mm(torch.mm(self.P_predict, H.t()), torch.inverse(psi))  # Kalman gain
#         P_plus = self.P_predict - torch.mm(torch.mm(K, psi), K.t())  # State error cov
#         state_plus = self.state_predict + torch.matmul(K, nu)

#         # Set the updated weights
#         self.state_minus = state_plus
#         self.set_param(state_plus)
#         self.P_minus = P_plus

#     def net_copy(self):
#         if self.type == 'MLP':
#             # optimize_copy = copy.deepcopy(self.optimize)
#             new_params = self.params
#         if self.type =='EKF':
#             P_copy = copy.deepcopy(self.P_minus)
#             q_copy = copy.deepcopy(self.params['q'])
#             r_copy = copy.deepcopy(self.params['r'])
#             new_params ={'type':self.type,'P':P_copy, 'q':q_copy,'r':r_copy}
#         return new_params

if __name__ == "__main__":
    optimizer = 'Adam'
    input_dim = 2
    output_dim = 7
    layer_wid = [8,128,output_dim]
    nonlinearity = 'relu'
    lr = .001
    params_mlp = {'type':'MLP','optimizer':optimizer,'lr':lr,'weight_decay':weight_decay}
    adam_net = Network(input_dim, layer_wid, nonlinearity, params_mlp)
    adam_net2 = copy.deepcopy(adam_net)
    P = '0.1'
    q = .00001
    r = 4
    params_ekf = {'type':'EKF','P':P,'q':q,'r':r}
    ekf_net = Network(input_dim, layer_wid, nonlinearity, params_ekf)

    # ekf_net2 = copy.deepcopy(ekf_net)
    print(ekf_net.state_dict)