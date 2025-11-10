import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# Dataset preparation
# transform = transforms.Compose([transforms.ToTensor()])
# train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
# test_data  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # pour les 3 canaux RGB
])
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# Visualize a few samples
if True:
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(6,2))
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.imshow(images[i][0], cmap="gray")
        plt.axis("off")
    plt.show()




class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(
        # self.conv2 = nn.Conv2d(
        # self.fc = nn.Linear( ??? , 10)

    def forward(self, x):
        # x = ...
        # x = ...
        # x = x.view(x.size(0), -1)
        # return self.fc(x)
        return x




# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         ...

#     def forward(self, x):
#         ...



# class CNN_SE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         ...

#     def forward(self, x):
#         ...
#         x = x.view(x.size(0), -1)
#         return self.fc(x)







def train_model(model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")



def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"(evaluation) Accuracy: {acc:.2f}%")
    return acc



if __name__ == "__main__":
    model = CNN_Base()
    print(model)
    #train_model(model, epochs=15)
    #evaluate(model)

    # se_model = CNN_SE()
    # print(se_model)
    # train_model(se_model, epochs=15)
    # evaluate(se_model)




