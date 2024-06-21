import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

def main() :
    mnist_train = datasets.MNIST(
        root='./MNIST/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    mnist_test = datasets.MNIST(
        root='./MNIST/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(
        dataset=mnist_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=mnist_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim):
            super(MLPClassifier, self).__init__()
            self.input_dim = input_dim

            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 32)
            self.fc5 = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x.view(-1, self.input_dim)))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    input_dim = mnist_train[0][0][0].shape[0] * mnist_train[0][0][0].shape[1]

    model = MLPClassifier(input_dim).to('cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def train(model, train_dataloader, criterion, optimizer):
        model.train()
        train_loss = []
        correct = 0
        total = 0

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        return sum(train_loss) / len(train_loss), correct / total

    def test(model, test_dataloader, criterion):
        model.eval()
        test_loss = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                data, labels = data.to('cpu'), labels.to('cpu')

                output = model(data)
                loss = criterion(output, labels)

                test_loss.append(loss.item())
                _, pred = torch.max(output, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            return sum(test_loss) / len(test_loss), correct / total

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        test_loss, test_acc = test(model, test_dataloader, criterion)

        print(
            f'Epoch {epoch}/{EPOCHS} - Train_Loss : {train_loss:.2f} Train_acc : {train_acc:.2f}'
            f' Test_Loss : {test_loss:.2f} Test_acc : {test_acc:.2f}'
        )

if __name__ == '__main__':
    main()