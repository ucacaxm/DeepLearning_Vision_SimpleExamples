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

img_size = 28
flat_dim = img_size * img_size

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
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, flat_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(-1, 1, img_size, img_size)  # ⬅ reshape correct

# ---------------------------------------------------------
# Modèle: Discriminateur
# ---------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)  # ⬅ flatten propre
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

        # -----------------------------
        # 1) Réelles images
        # -----------------------------
        real_imgs = imgs.to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)

        # -----------------------------
        # 2) Images générées
        # -----------------------------
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = G(z)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # -----------------------------
        # Train D
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
        # Train G
        # -----------------------------
        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, real_labels)  # generator wants D(fake)=1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss_D={loss_D.item():.4f}  Loss_G={loss_G.item():.4f}")

# ---------------------------------------------------------
# Génération finale
# ---------------------------------------------------------
z = torch.randn(16, latent_dim).to(device)
samples = G(z).cpu().detach()

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i][0], cmap="gray")  # canal 0 car (1,28,28)
    ax.axis("off")
plt.show()
