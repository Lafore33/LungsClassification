from graphs import show_losses
from run import run
import torch


def train(model, train_loader, test_loader, loss_function, optimizer, num_epoch):
    train_loss_hist = []
    test_loss_hist = []
    predictions = []
    min_loss = float('inf')
    for i in range(num_epoch):
        train_loss = run(model, train_loader, loss_function, optimizer)
        train_loss_hist.append(train_loss)
        test_loss, predictions = run(model, test_loader, loss_function)
        min_loss = min(min_loss, test_loss)
        test_loss_hist.append(test_loss)
        show_losses(train_loss_hist, test_loss_hist)
    path = './Model.pt'
    torch.save(model.state_dict(), path)

    return min_loss, predictions
