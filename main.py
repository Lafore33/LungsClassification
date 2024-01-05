from model import Model
from datasets import TrainDataset, TestDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from metrics import prepare_data
from sklearn.metrics import f1_score
from predict import predict
import pandas as pd
from training import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 30
batch_size = 32

data = TrainDataset(1)
test_dataset = TestDataset()

train_data, validation_data = torch.utils.data.random_split(data, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size)
validation_loader = DataLoader(validation_data, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

model = Model()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

min_loss, predictions = train(model, train_loader, validation_loader, loss_function, optimizer, NUM_EPOCHS)
y_true, pred = prepare_data(validation_loader, predictions)
print(f"f1_score = {f1_score(y_true, pred, average='weighted', zero_division=1)}")
print(f"min_loss = {min_loss}")

# loading the model
'''path = '/path/to/save'
model = Model()
model.load_state_dict(torch.load(path))'''

model.eval()
predictions = predict(test_dataloader, model)

# create a file with predictions
df = pd.DataFrame(predictions, columns=["target_feature"])
df.insert(loc=0, column='id', value=range(6920))
df.to_csv('Submission', index=False)