import matplotlib.pyplot as plt
import dlc_bci as bci
import numpy as np
import torch


from data_augmentation import interpolation
from neural_1d import Net_1D
from weight_initial import weights_init
from helper import *
from torch.autograd import Variable
from sklearn.model_selection import KFold

# import data from data set
train_input, train_target = bci.load(root='./data_bci')
test_input, test_target = bci.load(root='./data_bci', train=False)
# do sampling and interpolation
data_resample = interpolation(train_input, 50, 0)
train_input_new = np.concatenate((np.array(train_input), data_resample[0:200]), axis=0)
train_target_new = np.concatenate((np.array(train_target), np.array(train_target[0:200])), axis=0)
# 10-fold cross validation to split the data set into training and evaluation.
kf = KFold(n_splits=10, random_state=None, shuffle=True)
# convert test data into variable for final accuracy test.
test_input = torch.Tensor(test_input)
test_target = torch.LongTensor(test_target)
accuracy = 0
i = 0

for train_index, test_index in kf.split(train_input_new):
    i += 1
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_input_new[train_index], train_input_new[test_index]
    y_train, y_test = train_target_new[train_index], train_target_new[test_index]

    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.LongTensor(y_test)

    # Normalization
    mu, std = X_train.mean(), X_train.std()
    X_train.sub_(mu).div_(std)
    X_test.sub_(mu).div_(std)
    if i == 1:
       test_input.sub_(mu).div_(std)
       test_input, test_target = Variable(test_input), Variable(test_target)

    # Convert to Variable
    X_train, y_train = Variable(X_train), Variable(y_train)
    X_test, y_test = Variable(X_test), Variable(y_test)

    model = Net_1D()
    model.apply(weights_init)
    errors, losses, errors2 = train_model_early_stop(model, X_train, y_train, X_test, y_test)
    nb_error = compute_nb_errors(model, test_input, test_target)

    # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # ax[0].plot(np.array(errors)/X_train.size(0))
    # ax[0].plot(np.array(errors2)/X_test.size(0))
    # ax[0].set_title("Error Rate")
    # ax[0].grid()
    # ax[1].plot(np.array(losses))
    # ax[1].set_title("Loss")
    # ax[1].grid()
    # plt.show()

    print('Correct: {:.2%}'.format(1-nb_error/test_input.size(0)))
    accuracy += 1-nb_error/test_input.size(0)
    mean_accuracy = accuracy/15