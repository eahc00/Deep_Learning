import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore


def main() :
    wine_data_frame = pd.read_csv('winequality-white.csv', sep = ";")
    num_features = len(wine_data_frame.columns)
    print(wine_data_frame.isnull().sum())


    def preprocess_inputs(df, label_column):
        df = df.copy()

        # Split df into X and y
        Y = df[label_column]
        X = df.drop(label_column, axis=1)

        X = X.apply(zscore)

        Y = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, 1)
        X = torch.tensor(X.values, dtype=torch.float32)

        # Train-test split
        trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=1)

        return trainX, testX, trainY, testY

    trainX, testX, trainY, testY = preprocess_inputs(wine_data_frame, 'quality')


    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()

            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.dropout(x, 0.4, True)
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    input_dim = trainX.shape[1]

    model = MLP(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

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


    #%%
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()