import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import list_models
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import alexnet, vgg11, vgg19

from torchinfo import summary

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/vision/stable/models.html

if __name__ == '__main__':
    # classification_models = list_models(module=torchvision.models)
    # print("\n".join(classification_models))

    # summary(alexnet(), input_size=(1, 3, 224, 224), device='cpu')
    # summary(vgg11(), input_size=(1, 3, 224, 224), device='cpu')
    # summary(vgg19(), input_size=(1, 3, 224, 224), device='cpu')
    # summary(resnet18(), input_size=(1, 3, 224, 224), device='cpu')
    # exit(1)

    # transform = ResNet18_Weights.IMAGENET1K_V1.transforms()  # transform for ImageNet1K_V1 of resnet model. The input size will be 224x224

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32, 32)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    batch_size = 2

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


    class Net(nn.Module):
        def __init__(self, pretrained=False):
            super().__init__()
            if pretrained:
                self.base = resnet18(ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.base = resnet18()
            self.fc = nn.Linear(1000, 10)

        def forward(self, x):
            x = self.base(x)
            x = self.fc(x)
            return x

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    net = Net(pretrained=True)
    net.to(device)
    summary(net, input_size=(1, 3, 32, 32), device=device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')

    PATH = './cifar_resnet18.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images.to(device)
    labels = labels.to(device)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')