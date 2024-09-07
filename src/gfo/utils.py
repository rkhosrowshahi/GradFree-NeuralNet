import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
)
from ..models import *
from torchvision.models import *


def set_model_state(model, parameters):
    state = model.state_dict()
    counted_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = torch.tensor(
                parameters[counted_params : param.size().numel() + counted_params]
            ).reshape(param.size())
            counted_params += param.size().numel()

    model.load_state_dict(state)

    return model


def get_model_params(model):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(torch.flatten(param).cpu().detach().numpy())

    return np.concatenate(params)


def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    f1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, 28 * 28).to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            f1 += f1_score(
                target.view_as(pred).cpu().numpy(), pred.cpu().numpy(), average="macro"
            )

    return (
        total_loss / len(test_loader),
        correct / len(test_loader.dataset),
        f1 / len(test_loader),
    )


def get_network(net, in_size=None, hidden_size=None, num_classes=None):
    model = None
    if net == "resnet18":
        model = resnet18(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif net == "resnet34":
        model = resnet34(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif net == "resnet50":
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif net == "resnet101":
        model = resnet101(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif net == "resnet152":
        model = resnet152(weights="DEFAULT")
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif net == "alexnet":
        model = alexnet(weights="DEFAULT")
        dropout = 0.5
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    elif net == "lenet":
        model = LeNet(in_channels=1)
    elif net == "mlp":
        model = MLP(in_size=in_size, hidden_size=hidden_size, out_size=num_classes)
    elif net == "mnist-3M":
        model = MNIST3M()
    elif net == "mnist-500K":
        model = MNIST500K()
    elif net == "mnist-30K":
        model = MNIST30K()
    elif net == "cifar-8M":
        model = CIFAR8M()
    elif net == "cifar-900K":
        model = CIFAR900K()
    elif net == "cifar-300K":
        model = CIFAR300K()
    # elif net == "resnet18" and dataset == "cifar10":
    #     model = cresnet18(num_classes=10)
    #     model.load_state_dict(torch.load(model_path))
    # elif net == "resnet18" and dataset == "cifar100":
    #     model = cresnet18(num_classes=100)
    #     model.load_state_dict(torch.load(model_path))

    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    return model
