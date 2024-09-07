import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class MLP(nn.Module):
    def __init__(self, in_size=28 * 28, hidden_size=None, out_size=10):
        super(MLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if hidden_size is None:
            hidden_1 = in_size * in_size + 1
        else:
            hidden_1 = hidden_size
        self.fc1 = nn.Linear(in_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, out_size)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.in_size)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add output layer
        x = self.fc2(x)
        return x
