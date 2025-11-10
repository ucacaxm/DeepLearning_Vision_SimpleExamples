import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math


# =======================
# 1️ Hyperparamètres
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 300  # nombre d'étapes de diffusion
batch_size = 128
image_size = 32

# Plan de bruit linéaire (betas)
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, 0)


# =======================
# 2️ Fonctions diffusion
# =======================
def forward_diffusion(x0, t, noise=None):
    """Ajoute du bruit à l'image selon le pas t"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alphas_bar[t])[:, None, None, None]
    sqrt_1mab = torch.sqrt(1 - alphas_bar[t])[:, None, None, None]
    return sqrt_ab * x0 + sqrt_1mab * noise, noise


def get_index_from_list(vals, t, x_shape):
    """Renvoie un coeff de taille batch x 1 x 1 x 1"""
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# =======================
# 3️ U-Net simplifié avec attention
# =======================
class SelfAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, H*W)
        k = self.k(x).reshape(B, C, H*W)
        v = self.v(x).reshape(B, C, H*W)
        attn = torch.bmm(q.permute(0,2,1), k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v.permute(0,2,1)).permute(0,2,1).reshape(B, C, H, W)
        return x + self.proj(out)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.attn = SelfAttention2d(128)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t):
        t_embed = t[:, None, None, None].float() / T
        t_embed = t_embed.expand_as(x[:, :1, :, :])
        x = torch.cat([x, t_embed], dim=1)[:, :3]  # pour compatibilité simple
        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(F.avg_pool2d(x1, 2)))
        x2 = self.attn(x2)
        x3 = F.relu(self.up1(x2))
        x = self.out(x3 + x1)
        return x


# =======================
# 4️ Données CIFAR-10
# =======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


# =======================
# 5️ Entraînement
# =======================
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

for epoch in range(epochs):
    for imgs, _ in trainloader:
        imgs = imgs.to(device)
        t = torch.randint(0, T, (imgs.shape[0],), device=device).long()
        x_t, noise = forward_diffusion(imgs, t)
        noise_pred = model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")



# =======================
# 6️ Génération d'images
# =======================
def show_images(orig, recon, n=8):
    """
    Affiche les images originales et reconstruites côte à côte
    orig : batch original (B, C, H, W)
    recon : batch reconstruit (B, C, H, W)
    """
    orig_grid = torchvision.utils.make_grid(orig[:n], nrow=n, normalize=True, value_range=(-1,1))
    recon_grid = torchvision.utils.make_grid(recon[:n], nrow=n, normalize=True, value_range=(-1,1))

    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.title("Original Images")
    plt.imshow(orig_grid.permute(1,2,0).cpu())
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Reconstructed / Generated Images")
    plt.imshow(recon_grid.permute(1,2,0).cpu())
    plt.axis("off")
    plt.show()


# Prendre un batch d'images
imgs, _ = next(iter(trainloader))
imgs = imgs.to(device)

# Ajouter du bruit aléatoire à t fixe
t = torch.randint(0, T, (imgs.shape[0],), device=device).long()
x_t, _ = forward_diffusion(imgs, t)

# Reconstruire le bruit via le modèle
with torch.no_grad():
    noise_pred = model(x_t, t)
    recon = x_t - noise_pred  # reconstruction simple

# Visualiser
show_images(imgs, recon, n=8)

with torch.no_grad():
    samples = sample(model, n=8)  # sample() est la fonction reverse diffusion

show_images(samples, samples, n=8)  # on affiche juste les échantillons générés