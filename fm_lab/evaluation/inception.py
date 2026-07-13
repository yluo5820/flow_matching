"""TensorFlow-FID compatible Inception-v3 used by the reference evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

DEFAULT_FID_WEIGHTS = Path("stats/pt_inception-2015-12-05-6726825d.pth")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceInceptionV3(nn.Module):
    """Return 2,048-D pool features and FID-Inception probabilities."""

    def __init__(self, weights_path: str | Path = DEFAULT_FID_WEIGHTS) -> None:
        super().__init__()
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                "TensorFlow-FID Inception weights are required at "
                f"{weights_path}; torchvision ImageNet weights are not a compatible substitute."
            )
        inception = _fid_inception_v3(weights_path)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    inception.Conv2d_1a_3x3,
                    inception.Conv2d_2a_3x3,
                    inception.Conv2d_2b_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                ),
                nn.Sequential(
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d((1, 1)),
                ),
            ]
        )
        self.fc = inception.fc
        self.fc.bias = None
        self.weights_path = weights_path
        self.weights_sha256 = sha256_file(weights_path)
        self.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        x = 2.0 * x - 1.0
        for block in self.blocks:
            x = block(x)
        features = torch.flatten(x, 1)
        probabilities = F.softmax(self.fc(features), dim=1)
        return features, probabilities


def _fid_inception_v3(weights_path: Path) -> nn.Module:
    inception = torchvision.models.inception_v3(
        num_classes=1008,
        aux_logits=False,
        weights=None,
        init_weights=False,
    )
    inception.Mixed_5b = _FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = _FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = _FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = _FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = _FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = _FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = _FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = _FIDInceptionE1(1280)
    inception.Mixed_7c = _FIDInceptionE2(2048)
    inception.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return inception


class _FIDInceptionA(torchvision.models.inception.InceptionA):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3 = self.branch3x3dbl_3(
            self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        )
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat((branch1x1, branch5x5, branch3x3, branch_pool), dim=1)


class _FIDInceptionC(torchvision.models.inception.InceptionC):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        branch7x7dbl = self.branch7x7dbl_5(
            self.branch7x7dbl_4(
                self.branch7x7dbl_3(
                    self.branch7x7dbl_2(self.branch7x7dbl_1(x))
                )
            )
        )
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat((branch1x1, branch7x7, branch7x7dbl, branch_pool), dim=1)


class _FIDInceptionE1(torchvision.models.inception.InceptionE):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        base3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            (self.branch3x3_2a(base3x3), self.branch3x3_2b(base3x3)), dim=1
        )
        base3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            (
                self.branch3x3dbl_3a(base3x3dbl),
                self.branch3x3dbl_3b(base3x3dbl),
            ),
            dim=1,
        )
        branch_pool = self.branch_pool(
            F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )
        return torch.cat((branch1x1, branch3x3, branch3x3dbl, branch_pool), dim=1)


class _FIDInceptionE2(torchvision.models.inception.InceptionE):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        base3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            (self.branch3x3_2a(base3x3), self.branch3x3_2b(base3x3)), dim=1
        )
        base3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            (
                self.branch3x3dbl_3a(base3x3dbl),
                self.branch3x3dbl_3b(base3x3dbl),
            ),
            dim=1,
        )
        branch_pool = self.branch_pool(F.max_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat((branch1x1, branch3x3, branch3x3dbl, branch_pool), dim=1)
