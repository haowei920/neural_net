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
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        print(in_size)
        print(out_size)
        
        # self.fc1 = nn.Linear(in_size,32)
        # self.fc2 = nn.Linear(32,out_size)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_size)
        self.optimizer =  optim.Adam(self.parameters(),lr =lrate )
        # self.fc1 = nn.Linear() 




    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        for i in range(len(x)):
            mean = x[i].mean()
            std = x[i].std()
            x[i] = x[i] - mean
            x[i] = x[i] / std        
        # print(x[0])
        x = x.view((x.shape[0], 3, 32, 32))
        
        # print("local_x looked like this",x.size())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print("This is the current shape from my forward feed",x.size())
        return x
        # return x

    # def step(self, x,y):
    #     """
    #     Performs one gradient step through a batch of data x with labels y
    #     @param x: an (N, in_size) torch tensor
    #     @param y: an (N,) torch tensor
    #     @return L: total empirical risk (mean of losses) at this time step as a float
    #     """
    #     self.optimizer.zero_grad
    #     output = self.forward(x)
    #     loss = self.loss_fn(output,y)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item()


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
    """
    #https://www.youtube.com/watch?v=VZyTt1FvmfU&ab_channel=3Blue1Brown
    learning_rate = 0.001
    print("train set is of this type",type(train_set))
    print("train set is of this size",train_set.size())
    for indi in train_set:
        print("one item in the training set is of this size",indi.size())
        print("one item in training set is of this type",type(indi))
        break
    print("training set has this many data",train_set.shape[0])
    print("train label has this many data", train_labels.shape[0])
    # train_data = []

    # for i in range(len(train_set)):
    #     mean = train_set[i].mean()
    #     std = train_set[i].std()
    #     train_set[i] = train_set[i] - mean
    #     train_set[i] = train_set[i] / std

    # train_set = train_set.reshape(train_set.shape[0],3,32,32)
    # train_set = train_set.view((train_set.shape[0], 3, 32, 32))
    print("This is the type of new train set",type(train_set))
    print("This is the shape of new train set",train_set.size())

    # losses = [] 
    # net=NeuralNet(learning_rate,nn.CrossEntropyLoss(),train_set.shape[1],2)
    net=NeuralNet(learning_rate,nn.CrossEntropyLoss(),2,2)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001000)
    print("This is number of iter",n_iter)

    n_batches = int(len(train_set) / batch_size)
    # print("This is number of batches",int(n_batches))
    # shuffled_idx =((np.arange(len(train_set))))
    shuffled_idx = np.arange(len(train_set))
    np.random.shuffle(shuffled_idx)

    #n_iter is 500, batch_size is 100
    for epoch in range(1):
        losses = [] 
        for i in range(n_iter):
            # print(i)
            if (i + 1) % n_batches == 0:
                batch_idx = shuffled_idx[((i-1) % (n_batches))*batch_size: ((i) %(n_batches)) * batch_size]
            else:
                batch_idx = shuffled_idx[(i % (n_batches))*batch_size: ((i+1) %(n_batches)) * batch_size]
            # local_x = train_set[(i)*batch_size:((i+1))*batch_size]
            # local_y = train_labels[i*batch_size:(i+1)*batch_size]

            local_x = train_set[batch_idx]
            
            # print("local_x looked like this",local_x.size())
            local_y = train_labels[batch_idx]
            net.zero_grad()
            output=net.forward(local_x)
            
            loss = net.loss_fn(output,local_y)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            # losses.append(int(loss))
           
           

        # losses.append(loss_total)




    # print(loss)
    # for i in range(len(dev_set)):
    #     mean = dev_set[i].mean()
    #     std = dev_set[i].std()
    #     dev_set[i] = dev_set[i] - mean
    #     dev_set[i] = dev_set[i] / std
    # print("This is the original dimension for development set",dev_set.size())
    # dev_set = dev_set.reshape(dev_set.shape[0],3,32,32)
    print("This is the new_dev_set size",dev_set.size())
    matrix = net.forward(dev_set)
    print("This is matrix",matrix)
    print("This is matrix dimension",matrix.size())
    # matrix = net.forward(dev_set)
    yhats = np.argmax(matrix.detach().numpy(), axis=1)
    print("This is my yhats for all the dev data",yhats)
    print("This is the shape of yhats for all the dev data",yhats.shape)
    print(losses)
    print("This is length of losses",len(losses))
    return losses,yhats,net