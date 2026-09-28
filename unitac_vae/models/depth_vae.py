"""Depth-map VAE: pre-trained on contact depth maps, it defines the shared target latent space.

Input/output: (B, 1, 50, 50) z-score-normalised depth maps. Submodule names are kept
identical to the original experiment code so archived state dicts load unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import GRID_SIZE


class DepthVAEEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1), nn.ReLU(True), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(True), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True), nn.BatchNorm2d(128),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class DepthVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, output_channels: int = 1, grid_size: int = GRID_SIZE):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(True), nn.BatchNorm2d(16),
            nn.Upsample(size=grid_size, mode="bilinear", align_corners=False),
            nn.Conv2d(16, output_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.decoder(self.decoder_input(z).view(z.size(0), 128, 4, 4))


class DepthVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, input_channels: int = 1, output_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = DepthVAEEncoder(input_channels, latent_dim)
        self.decoder = DepthVAEDecoder(latent_dim, output_channels)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

