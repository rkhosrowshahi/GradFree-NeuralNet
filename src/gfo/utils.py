import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
)


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
