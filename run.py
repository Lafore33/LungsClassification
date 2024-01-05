import torch
from datasets import augmenting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(model, dataloader, loss_function, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0
    predictions = []

    for X, y in dataloader:
        if optimizer is not None:
            X = augmenting(X)
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)
        total_loss += loss.item()
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            predictions.append(pred.argmax(dim=1))

    return total_loss / len(dataloader), predictions if optimizer is None else total_loss / len(dataloader)
