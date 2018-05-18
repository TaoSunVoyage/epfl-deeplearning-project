import dlc_bci as bci
import numpy as np
import torch
from torch.autograd import Variable

from neural_1d import Net_1D
from weight_initial import weights_init
from data_augmentation import *
from helper import *

# Import data
train_input_original, train_target_original = bci.load(root='./data_bci')
test_input, test_target = bci.load(root='./data_bci', train=False)

# Data augmentation by sampling and interpolation to form new data
data_resample = interpolation(train_input_original, 50, 0)
train_input = np.concatenate((np.array(train_input_original), data_resample[0:200]), axis=0)
train_input = torch.Tensor(train_input)
train_target = np.concatenate((np.array(train_target_original), np.array(train_target_original[0:200])), axis=0)
train_target = torch.LongTensor(train_target)

# Normalization
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

# Convert to Variable
train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)


# For neural network
train_input = train_input.view(516, 28, 50)
test_input = test_input.view(100, 28, 50)


# Apply the model 10 times
# Calculate the errors and loss
correct_list = []
for i in range(10):
    print("*"*30)
    print("No.{}".format(i+1))
    model = Net_1D()
    # Apply weight initialization
    model.apply(weights_init)
    errors, losses = train_model(model, train_input, train_target)
    nb_error = compute_nb_errors(model, test_input, test_target)
    correct = 1 - nb_error / test_input.size(0)
    correct_list.append(correct)
    print('Correct: {:.2%}'.format(1-nb_error/test_input.size(0)))

# Mean accuracy calculation
print('The mean of accuracy: {:.2%}'.format(np.mean(correct_list)))
