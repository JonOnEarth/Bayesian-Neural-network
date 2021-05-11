"""
check the partial BNN paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm


class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.Tensor(self.output_features, self.input_features).uniform_(-0.05, 0.05))
        self.w_rho = nn.Parameter(torch.Tensor(self.output_features, self.input_features).uniform_(-2, -1))
        
        # self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        # self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        # #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.Tensor(self.output_features).uniform_(-0.05, 0.05))
        self.b_rho = nn.Parameter(torch.Tensor(self.output_features).uniform_(-2, -1))

        # self.b_mu =  nn.Parameter(torch.zeros(output_features))
        # self.b_rho = nn.Parameter(torch.zeros(output_features))        

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)


    def forward(self, x):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0,1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        
        return F.linear(x, self.w, self.b)


    def get_param(self):
        """
        Return weights of this net as a vector
        """
        param_vector = torch.tensor([])  # Initial weights
        for param in self.parameters():
            param_vector = torch.cat((param_vector, param.view(-1)))

        return param_vector

class MLP_BBB(nn.Module):
    def __init__(self, input_dim, layer_wid, bnn_layer_id, nonlinearity, noise_tol=.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        
        self.fc_layers = nn.ModuleList()
        self.bnn_layer_id = bnn_layer_id

        # self.fc_layers.append(Linear_BBB(input_dim,layer_wid[0], prior_var=prior_var))
        self.fc_layers.append(nn.Linear(input_dim,layer_wid[0], bias=True))
        for i in range(len(layer_wid) -1):
            if i+1 in bnn_layer_id:
                self.fc_layers.append(Linear_BBB(layer_wid[i],layer_wid[i + 1], prior_var=prior_var))
            else:
                self.fc_layers.append(nn.Linear(layer_wid[i],layer_wid[i + 1]))
        # self.out = Linear_BBB(hidden_units, 1, prior_var=prior_var)
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

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

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        for fc_layer in self.fc_layers[:-1]:
            x = self.nonlinearity(fc_layer(x))
        # x = torch.sigmoid(self.hidden1(x))
        # x = torch.sigmoid(self.hidden2(x))
        # x = self.out(x)
        return self.fc_layers[-1](x)

    def log_prior(self):
        # calculate the log prior over all the layers
        log_priors = 0

        for i, fc_layer in enumerate(self.fc_layers):
            if i in [j+1 for j in self.bnn_layer_id]:
                log_priors = log_priors + fc_layer.log_prior
        # return self.hidden1.log_prior + self.hidden2.log_prior + self.out.log_prior
        return log_priors

    def log_post(self):
        # calculate the log posterior over all the layers
        log_posts = 0
        for i, fc_layer in enumerate(self.fc_layers):
            if i in [j+1 for j in self.bnn_layer_id]:
                log_posts = log_posts + fc_layer.log_post
        return log_posts

        # return self.hidden1.log_post + self.hidden2.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0]*target.shape[1])   
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss

     
    def sample_elbo_fed(self, input, target, samples):
        '''
        # copy sample_elbo function to use in federated learning
        for every worker, we need to transmit the gradient of log like y(D_k,n|w) to center
        '''
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0]*target.shape[1])   
        
        log_likes = torch.zeros(samples)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss_pq = log_post - log_prior 
        loss_like = - log_like
        
        return loss_pq, loss_like


def main():
    nonlinearity2 =  'leaky_relu'
    global_model_bnn = MLP_BBB(input_dim,layer_wid2, nonlinearity2, noise_tol = 0.1, prior_var=1)
    optimizer2 = optim.Adam(bnet.parameters(), lr=.01, weight_decay = 1e-5)

    epochs2 = 2000
    for epoch in range(epochs2):  # Loop over the dataset multiple times
        losses, train_mses = [], []
        loss_likes = 0
        for idx in idxs_users:
            local_model = copy.deepcopy(global_model_bnn)
            ids = list(dict_users[idx]) # dataset ids
            
            loss_like = local_model.sample_elbo_fed(train_x[ids], train_y[ids], samples)
            loss_likes += loss_like

        loss = global_model_bnn.kl_pq(samples) + loss_likes
        loss.backward()
        optimizer2.step()
        losses.append(loss.item())

        output_train = global_model_bnn(train_x)
        train_mse = F.mse_loss(train_y, output_train)
        train_mses.append(train_mse.item())
        

        test_mses = []
        test_pred = bnet(x_test)
        test_mse = F.mse_loss(y_test, test_pred)
        test_mses.append(test_mse.item())

        print(f"train_mse:{train_mse},test_mse:{test_mse}")
        
    return losses, mses

