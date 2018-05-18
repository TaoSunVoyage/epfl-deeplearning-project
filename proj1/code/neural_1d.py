from torch import  nn
from torch.nn import functional as F


class Net_1D(nn.Module):
    def __init__(self):
        super(Net_1D, self).__init__()

        self.conv1 = nn.Conv1d(28, 50, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm1d(50)
        self.conv1_drop = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(50, 80, kernel_size=4, padding=2)
        self.conv2_bn = nn.BatchNorm1d(80)
        self.conv2_drop = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(80, 160, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm1d(160)
        self.conv3_drop = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(160, 160, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm1d(160)
        self.conv4_drop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(800, 300)
        self.fc1_bn = nn.BatchNorm1d(300)

        self.fc2 = nn.Linear(300, 30)
        self.fc2_bn = nn.BatchNorm1d(30)

        self.fc3 = nn.Linear(30, 30)
        self.fc3_bn = nn.BatchNorm1d(30)

        self.fc4 = nn.Linear(30, 30)
        self.fc4_bn = nn.BatchNorm1d(30)

        self.fc5 = nn.Linear(30, 2)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool1d(x, 3)
        x = self.conv1_drop(x)

        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool1d(x, 3)
        x = self.conv2_drop(x)

        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = self.conv3_drop(x)

        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = self.conv4_drop(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1_bn(self.fc1(x)))

        x = F.relu(self.fc2_bn(self.fc2(x)))

        x1 = x

        x = F.elu(self.fc3_bn(self.fc3(x)))

        x = F.elu(self.fc4_bn(self.fc4(x)))

        # Residual Network
        x = F.softmax(self.fc5(x+x1), 1)

        return x
