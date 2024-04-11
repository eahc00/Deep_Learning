import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore

from sklearn.datasets import fetch_california_housing

if __name__=='__main__':
    housing = fetch_california_housing()

    data = pd.DataFrame(housing.data)
    data.columns = housing.feature_names
    num_features = len(housing.feature_names)

    print(data.isnull().sum())

    # feature normalization
    # min-max normalizing (from 0 to 1)
    # data = (data - data.min()) / (data.max() - data.min())
    # z-score normalizing
    data = data.apply(zscore)

    # convert into tensors
    x_values = data.values
    x_values = torch.tensor(x_values, dtype=torch.float32)
    y_values = torch.tensor(housing.target, dtype=torch.float32).reshape(-1, 1)

    trainX, testX, trainY, testY = train_test_split(x_values, y_values, random_state=0, test_size=0.2)

    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()

            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_dim = x_values.shape[1]
    model = MLP(input_dim)

    # loss
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    losses = []

    # training the model
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(trainX)
        loss = criterion(outputs, trainY)
        loss.backward()

        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs} - Loss : {loss.item():.2f}')

        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(testX)
        test_loss = criterion(test_outputs, testY)
        print(f'Test Loss : {test_loss.item():.2f}')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()
