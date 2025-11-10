import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# Dataset preparation
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# Visualize a few samples
if False:
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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64*7*7, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        return self.fc(x)




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # Squeeze
        self.fc = nn.Sequential(                    # Excitation
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()       # b=batch size, c = chanel size
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y 



class CNN_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock(64)
        self.fc = nn.Linear(64*7*7, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.se1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.se2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)







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
    train_model(model, epochs=5)
    evaluate(model)

    se_model = CNN_SE()
    print(se_model)
    train_model(se_model, epochs=5)
    evaluate(se_model)

    x, _ = next(iter(test_loader))
    _ = se_model(x[:1])  # Forward pass
    with torch.no_grad():
        se_output = se_model.se1.avg_pool(x[:1]).view(-1)
        plt.bar(range(len(se_output)), se_output.numpy())
        plt.title("Channel-wise importance (SE Block 1)")
        plt.xlabel("Channels")
        plt.ylabel("Weight")
        plt.show()


