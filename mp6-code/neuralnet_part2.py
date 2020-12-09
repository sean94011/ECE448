# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn

class Conv2Lin(nn.Module):
    def forward(self,x):
        # print(x.view(x.shape[0],-1).shape)
        # print(x.shape)
        return x.view(x.shape[0],-1)

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Conv2Lin(),
            nn.Linear(4096,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,out_size)
        )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lrate, weight_decay=0.001)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        standardized_x = (x - torch.mean(x))/torch.std(x)
        standardized_x = (standardized_x.view(-1, 3, 32, 32))#.cuda()
        return self.model(standardized_x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        y_hat = self(x)
        loss = self.loss_fn(y_hat,y)
        loss.backward()
        self.optimizer.step()
        return loss



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(0)
    #     print(torch.cuda.get_device_name(0))
    standardized_dev = (np.subtract(dev_set , torch.mean(dev_set))/torch.std(dev_set))
    
    loss_fn = nn.CrossEntropyLoss()
    lrate = 0.03
    in_size = train_set.shape[1]
    out_size = 2
    net = NeuralNet(lrate,loss_fn,in_size,out_size)#.cuda()

    losses = []

    for i in range(n_iter):
        start_index = i*batch_size % train_set.shape[0]
        x_batch = train_set[start_index:start_index+batch_size]
        y_batch = (train_labels[start_index:start_index+batch_size])#.cuda()
        net.train()
        loss = net.step(x_batch,y_batch)
        losses.append(loss.item())

    net.eval()
    y_hat = np.zeros(dev_set.shape[0])
    with torch.no_grad():
        for i in range(dev_set.shape[0]):
            image = standardized_dev[i]
            output = net(image)
            prediction = torch.argmax(output)
            y_hat[i] = prediction.item()

    return losses,y_hat,net
