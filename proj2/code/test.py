"""
Test file of our framework.

Author: Tao Sun, Wenlong Deng, Yaxiong Luo
"""

from module import *
from torch import FloatTensor, LongTensor
import math
import numpy as np


def reshapeLabel(label):
    """
    Reshape 1-D [0,1,...] to 2-D [[1,-1],[-1,1],...].
    """
    n = label.size(0)
    y = FloatTensor(n, 2)
    y[:, 0] = 2 * (0.5 - label)
    y[:, 1] = - y[:, 0]
    return y.long()


# Generate Data points
X_train = FloatTensor(1000, 2).uniform_(0, 1)
y_label_train = X_train.sub(0.5).pow(2).sum(1).lt(1./2./math.pi).float()
y_train = reshapeLabel(y_label_train)

X_test = FloatTensor(1000, 2).uniform_(0, 1)
y_label_test = X_test.sub(0.5).pow(2).sum(1).lt(1./2./math.pi).float()
y_test = reshapeLabel(y_label_test)


# Normalization
mu, std = X_train.mean(), X_train.std()
X_train.sub_(mu).div_(std)
X_test.sub_(mu).div_(std)


# Build neural network
model = Sequential()
model.add(Linear(2, 25), ReLU(),
          Linear(25, 25), ReLU(),
          Linear(25, 25), ReLU(),
          Linear(25, 2), Tanh())


# Parameters setting
lr = 0.01
mini_batch_size = 50
epoch = 10

criterion = LossMSE()

# Training
for e in range(epoch):
    print("******** Begin Epoch {} ********\n".format(e))
    for b in range(0, X_train.size(0), mini_batch_size):
        # Forward propagation
        output = model.forward(X_train.narrow(0, b, mini_batch_size))
        # Loss calculation
        loss = criterion.forward(output, y_train.narrow(0, b, mini_batch_size).float())
        # Error calculation
        error = output.max(1)[1].ne(y_train.narrow(0, b, mini_batch_size).max(1)[1]).sum()/output.size(0)
        # Print loss and error
        print("Epoch {} Batch {:2.0f}: {:4.2f}, {:6.2%}"
              .format(e, b/mini_batch_size, loss, error))
        # Backward propagation
        # Calculate the gradient of loss w.r.t network output
        l_grad = criterion.backward()
        # Calculate gradients in the network
        model.backward(l_grad)
        # Update parameters
        for layer in model.param():
            for p in layer:
                p[0].sub_(lr*p[1])

    print("\n******** After Epoch {} ********\n".format(e))
    output = model.forward(X_train)
    loss = criterion.forward(output, y_train.float())
    error = output.max(1)[1].ne(y_train.max(1)[1]).sum()/output.size(0)
    print("Loss: {:4.2f}, Error: {:6.2%}\n".format(loss, error))


# After training
# Training error
train_output = model.forward(X_train)
print("Train Error: {:.2%}".format(train_output.max(1)[1].ne(y_train.max(1)[1]).sum()/train_output.size(0)))
# Test error
test_output = model.forward(X_test)
print("Test Error: {:.2%}".format(test_output.max(1)[1].ne(y_test.max(1)[1]).sum()/test_output.size(0)))
