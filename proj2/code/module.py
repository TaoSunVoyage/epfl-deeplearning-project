"""
Modules of our framework.

Author: Tao Sun, Wenlong Deng, Yaxiong Luo
"""

from torch import FloatTensor, LongTensor
import math


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear).__init__()

        self.name = 'Linear'

        self.in_features = in_features
        self.out_features = out_features

        self.weight = FloatTensor(in_features, out_features)
        self.weight_grad = FloatTensor(in_features, out_features)

        if bias:
            self.bias = FloatTensor(out_features)
            self.bias_grad = FloatTensor(out_features)
        else:
            self.bias = None
            self.bias_grad = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.uniform_(-stdv, stdv)
        self.weight_grad.fill_(0)
        if self.bias is not None:
            self.bias.uniform_(-stdv, stdv)
            self.bias_grad.fill_(0)

    def forward(self, input):
        # Y = X * W + b
        self.input = input.clone()
        if self.bias is not None:
            return self.input.matmul(self.weight).add(self.bias)
        else:
            return self.input.matmul(self.weight)

    def backward(self, gradwrtoutput):
        # dW = X^T * dL/dY
        self.weight_grad = self.input.t().matmul(gradwrtoutput)
        # db = (dL/dY)^T * 1
        if self.bias is not None:
            self.bias_grad = gradwrtoutput.t().sum(1)
        # dX = dL/dY * W^T
        return gradwrtoutput.matmul(self.weight.t())

    def param(self):
        if self.bias is not None:
            return [(self.weight, self.weight_grad),
                    (self.bias, self.bias_grad)]
        else:
            return [(self.weight, self.weight_grad)]


class ReLU(Module):
    def __init__(self):
        super(ReLU).__init__()

        self.name = 'ReLU'

    def forward(self, input):
        # Y = max(0,X)
        self.input = input.clone()
        return self.input.mul(self.input.gt(0).float())

    def backward(self, gradwrtoutput):
        if self.input is not None:
            return gradwrtoutput.mul(self.input.gt(0).float())
        else:
            print("Forward First")
            return None

    def param(self):
        return []


class Tanh(Module):
    def __init__(self):
        super(Tanh).__init__()
        self.name = 'Tanh'

    def forward(self, input):
        # Y = (exp(X) - exp(-X))/(exp(X) + exp(-X))
        self.input = input.clone()
        return self.input.tanh()
        # self.input.masked_fill_(self.input.gt(50), 50)  # prevent overflow
        # return (self.input.exp() - self.input.mul(-1).exp()) / (self.input.exp() + self.input.mul(-1).exp())

    def backward(self, gradwrtoutput):
        # dY/dX = 4/(exp(x) + exp(-x))^2
        if self.input is not None:
            grad = 4. / (self.input.exp() + self.input.mul(-1).exp()).pow(2)
            return gradwrtoutput.mul(grad)
        else:
            print("Forward First")
            return None

    def param(self):
        return []


class Sequential(Module):
    def __init__(self):
        super(Sequential).__init__()
        self.name = 'Sequential'
        self.module_list = []

    def add(self, *module):
        for m in module:
            self.module_list.append(m)

    def forward(self, input):
        module_input = input.clone()
        for module in self.module_list:
            module_output = module.forward(module_input)
            module_input = module_output
        return module_output

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput
        for module in self.module_list[::-1]:
            grad = module.backward(grad)
        # return grad

    def param(self):
        return [module.param() for module in self.module_list]


class LossMSE():
    def __init__(self):
        super(LossMSE).__init__()
        self.name = 'LossMSE'

    def forward(self, predict_value, true_value):
        self.predict_value = predict_value
        self.true_value = true_value
        self.loss = (self.predict_value - self.true_value).float().pow(2).mean()
        return self.loss

    def backward(self):
        return 2 * (self.predict_value - self.true_value).float()
