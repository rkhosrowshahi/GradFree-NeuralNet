import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 64
        # hidden_2 = 32
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        # self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_1, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add hidden layer, with relu activation function
        # x = F.relu(self.fc2(x))
        # add output layer
        x = self.fc3(x)
        return x
