from torch import nn


def predict(test_dataloader, model):
    predictions = []
    for images in test_dataloader:
        raw_predictions = model(images)
        updated_predictions = nn.Softmax(1)(raw_predictions)
        predictions.extend(updated_predictions.argmax(1))

    for i in range(6920):
        predictions[i] = int(predictions[i])

    return predictions
