import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset preparation
# transform = transforms.Compose([transforms.ToTensor(),])
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64*8*8, 10)     # 64 features/channels, 7x7 image size after pooling (28/2/2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(x.size(0), -1)       # flatten, x.size(0) is the batch size
        return self.fc(x)




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # Squeeze
        self.fc = nn.Sequential(                    # Excitation
            nn.Linear(channels, channels // reduction, bias=False),     #channels // reduction
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()       # b=batch size, c = chanel size
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y                    # Scale the input feature maps by the channel-wise weights



class CNN_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock(64)
        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        # x = self.se1(F.relu(self.conv1(x)))
        # x = F.max_pool2d(x, 2)
        # x = self.se2(F.relu(self.conv2(x)))
        # x = F.max_pool2d(x, 2)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.se1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.se2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)




class CNN_MHA(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Multihead Attention parameters
        self.num_heads1 = 4
        self.num_heads2 = 4
        self.mha1 = nn.MultiheadAttention(embed_dim=32, num_heads=self.num_heads1, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=64, num_heads=self.num_heads2, batch_first=True)
        
        # Fully connected
        self.fc = nn.Linear(64*8*8, num_classes)

    def forward(self, x):
        # --- Bloc 1 ---
        x = F.relu(self.conv1(x))  # (B, 32, 28, 28)
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h*w).permute(0, 2, 1)  # (B, N, C)
        x_attn, _ = self.mha1(x_flat, x_flat, x_flat)  # (B, N, C)
        x = x_attn.permute(0, 2, 1).view(b, c, h, w)
        x = F.max_pool2d(x, 2)  # (B, 32, 14, 14)

        # --- Bloc 2 ---
        x = F.relu(self.conv2(x))  # (B, 64, 14, 14)
        b, c, h, w = x.size()
        x_flat = x.view(b, c, h*w).permute(0, 2, 1)  # (B, N, C)
        x_attn, _ = self.mha2(x_flat, x_flat, x_flat)  # (B, N, C)
        x = x_attn.permute(0, 2, 1).view(b, c, h, w)
        x = F.max_pool2d(x, 2)  # (B, 64, 7, 7)

        # --- Classifier ---
        #x = x.view(b, -1)
        x = x.reshape(b, -1)
        return self.fc(x)







def train_model(model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")



def evaluate(model):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x,y = x.to(device), y.to(device)
            preds = model(x)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"(evaluation) Accuracy: {acc:.2f}%")
    return acc



if __name__ == "__main__":
    test = [ False, False, True ]
    #test = [ True, True, True ]
    ep = 15
    if test[0]:
        model = CNN_Base()
        #print(model)
        train_model(model, epochs=ep)
        evaluate(model)

    if test[1]:
        se_model = CNN_SE()
        #print(se_model)
        train_model(se_model, epochs=ep)
        evaluate(se_model)

    if test[2]:
        mha_model = CNN_MHA()
        #print(mha_model)
        train_model(mha_model, epochs=ep)
        evaluate(mha_model)


    # x, _ = next(iter(test_loader))
    # x = x.to(device)
    # _ = se_model(x[:1])  # Forward pass
    # with torch.no_grad():
    #     # Forward jusqu'à obtenir le "channel weights" de SE Block 1
    #     y = se_model.se1.avg_pool(x[:1])      # shape (1, C, 1, 1)
    #     y = y.view(1, -1)                     # shape (1, C)
    #     y = se_model.se1.fc[:2](y)            # première partie du MLP (Linear+ReLU)
    #     y = se_model.se1.fc[2:](y)            # deuxième Linear + Sigmoid
    #     se_output = se_model.se1.fc(y).view(-1) # ou simplement output après fc

    #     plt.bar(range(len(se_output)), se_output.cpu().numpy())
    #     plt.title("Channel-wise importance (SE Block 1)")
    #     plt.xlabel("Channels")
    #     plt.ylabel("Weight")
    #     plt.show()


