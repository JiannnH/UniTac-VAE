"""Sensor-specific VAEs whose latent distributions are aligned to the depth-map VAE's.

Vision sensors (GelSight Mini, DIGIT) use a 4-block CNN on (3, 320, 240) images.
Taxel sensors use MLPs: PapillArray on 27-d (9 pillars x fx,fy,fz), uSkin on 72-d
(24 taxels x x,y,z). All operate on z-score-normalised signals.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..constants import IMAGE_HW, VISION_SENSORS


class VisionVAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VisionVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, image_hw: Tuple[int, int] = IMAGE_HW):
        super().__init__()
        h, w = image_hw[0] // 16, image_hw[1] // 16
        self.decoder_input = nn.Linear(latent_dim, 256 * h * w)
        self.decoder_net = nn.Sequential(
            nn.Unflatten(1, (256, h, w)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, z):
        return self.decoder_net(self.decoder_input(z))


class PapillEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(27, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)


class PapillDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.output_shape = (9, 3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 27),
        )

    def forward(self, z):
        return self.decoder(z).view(-1, *self.output_shape)


class XelaEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(72, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.encoder(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)


class XelaDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.output_shape = (24, 3)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 72),
        )

    def forward(self, z):
        return self.decoder(z).view(z.size(0), *self.output_shape)


class SensorVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

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


def build_sensor_vae(sensor: str, latent_dim: int = 64, image_hw: Tuple[int, int] = IMAGE_HW) -> SensorVAE:
    if sensor in VISION_SENSORS:
        return SensorVAE(VisionVAEEncoder(latent_dim), VisionVAEDecoder(latent_dim, image_hw))
    if sensor == "papill":
        return SensorVAE(PapillEncoder(latent_dim), PapillDecoder(latent_dim))
    if sensor == "xela":
        return SensorVAE(XelaEncoder(latent_dim), XelaDecoder(latent_dim))
    raise ValueError(f"Unknown sensor {sensor!r}")
