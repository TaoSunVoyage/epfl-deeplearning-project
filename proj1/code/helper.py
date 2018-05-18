import torch
import numpy as np
from torch import optim
from torch.optim import lr_scheduler


def train_model(model, train_input, train_target):
    """
    Train model.
    """
    lr, nb_epochs = 2e-3, 150
    # in the optimizer, the weight decay is the weight of regulization part.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.225)
    # the number of patience means Number of epochs with no improvement after which learning rate will be reduced
    # Factor by which the learning rate will be reduced. new_lr = lr * factor.
    # Threshold for measuring the new optimum, to only focus on significant changes.
    # Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=22, factor=0.02, threshold=1e-4, eps=1e-9)

    criterion = torch.nn.CrossEntropyLoss()
    errors = []
    losses = []
    error = np.inf

    print("Epochs   Error       Loss")
    for k in range(nb_epochs):
        scheduler.step(error)
        optimizer.zero_grad()
        model_train = model.train()
        output = model_train(train_input)
        loss = criterion(output, train_target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        error = pred.ne(train_target.data.view_as(pred)).sum()
        if k % 10 == 0:
            print("{}     {}     {}".format(k, error, loss.data[0]))
        errors.append(error)
        losses.append(loss.data[0])
    return errors, losses


def compute_nb_errors(model, data_input, data_target):
    """
    Compute errors.
    """
    nb_data_errors = 0
    model_eval = model.eval()
    output = model_eval(data_input)
    _, predicted_classes = torch.max(output, 1)
    for k in range(len(data_target)):
        if data_target.data[k] != predicted_classes.data[k]:
            nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def train_model_early_stop(model, train_input, train_target, X_test, y_test):
    """
    Training model with early stop criterion.
    """
    lr, nb_epochs = 2e-3, 150
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.24)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=22, factor=0.02, threshold=1e-4, eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss()
    train_error_list = []
    eval_error_list = []
    losses = []
    metric = np.inf
    gap = 1000
    count = 0
    for k in range(nb_epochs):
        scheduler.step(metric)
        optimizer.zero_grad()
        model_train = model.train()
        output = model_train(train_input)
        loss = criterion(output, train_target)
        loss.backward()
        optimizer.step()

        train_pred = output.data.max(1, keepdim=True)[1]
        # this error is training error
        train_error = train_pred.ne(train_target.data.view_as(train_pred)).sum()
        eval_pred = model.forward(X_test).data.max(1, keepdim=True)[1]
        # the error is evaluation error
        eval_error = eval_pred.ne(y_test.data.view_as(eval_pred)).sum()
        # early stop is to stop the training step earlier
        # if the gap between evaluation error and training error increases twice successively

        # save the smallest gap between training and evaluation error
        if gap > eval_error-train_error:
           gap = eval_error-train_error
        # if the gap between them larger than the samllest gap two consecutive times we stop the training.
        if eval_error-train_error > gap:
            count += 1
            if count > 2:
                break
        if k % 10 == 0:
            print(k, train_error, loss.data[0])
        train_error_list.append(train_error)
        eval_error_list.append(eval_error)
        losses.append(loss.data[0])
    return train_error_list, losses, eval_error_list
