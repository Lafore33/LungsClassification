import numpy as np


def prepare_data(test_loader, predictions):
    y_true = []
    pred = []
    for X, y in test_loader:
        y_true.extend(np.array(y))
    for i in range(len(predictions)):
        pred.extend(predictions[i].cpu().numpy())
    return y_true, pred
