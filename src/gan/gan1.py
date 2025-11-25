import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------
latent_dim = 100
batch_size = 128
lr = 0.0002
epochs = 30
img_size = 28*28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Chargement de MNIST
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------
# Modèle: Générateur
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.net(z)
        return img

# ---------------------------------------------------------
# Modèle: Discriminateur
# ---------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)

# ---------------------------------------------------------
# Initialisation
# ---------------------------------------------------------
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr)
opt_D = optim.Adam(D.parameters(), lr=lr)

# ---------------------------------------------------------
# Entraînement
# ---------------------------------------------------------
for epoch in range(epochs):
    for imgs, _ in loader:

        # Vraies images
        real_imgs = imgs.view(imgs.size(0), -1).to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)

        # Images générées
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = G(z)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # -----------------------------
        # 1) Entraîner le discriminateur
        # -----------------------------
        D_real = D(real_imgs)
        loss_real = criterion(D_real, real_labels)

        D_fake = D(fake_imgs.detach())
        loss_fake = criterion(D_fake, fake_labels)

        loss_D = (loss_real + loss_fake) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # -----------------------------
        # 2) Entraîner le générateur
        # -----------------------------
        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs}  |  Loss_D = {loss_D.item():.4f}  |  Loss_G = {loss_G.item():.4f}")

# ---------------------------------------------------------
# Génération finale
# ---------------------------------------------------------
z = torch.randn(16, latent_dim).to(device)
samples = G(z).view(-1, 28, 28).cpu().detach()

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i], cmap="gray")
    ax.axis("off")
plt.show()
