from torch import nn


def weights_init(m):
    """
    Initialize weight of CNN.
    """
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
