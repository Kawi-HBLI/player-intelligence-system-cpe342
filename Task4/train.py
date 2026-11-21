"""Task 4: game title detection from screenshots (5 classes, macro F1).

Trains on the GPU with all training images held in RAM and augmentation done on
the GPU, and records timing, per-class scores and the confusion matrix.
Options added per round: 1 --mode leak-check; 2 --split/--aug/--schedule/
--label-smoothing/--class-weight; 3 --model/--fold/--micro-batch/--resume.
--mode predict labels the test images with a trained run.

Run from the repository root with a CUDA build of PyTorch (best setting so far, ~1 h):
    python Task4/train.py --run-name r3d_lr2e-4 --model convnext_tiny --split near-copy
        --epochs 15 --schedule cosine --aug strong --lr-backbone 2e-4
    python Task4/train.py --mode predict --run-name r3d_lr2e-4
The second command writes outputs/main/task4_predictions.csv for `python run.py assemble`.

Rules followed here: training, validation and every setting use training images
only. Test images are opened only by --mode predict, to label them; no statistic
of them feeds back into training, the split, augmentation or thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torchvision import models

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parent


def default_data_dir() -> Path:
    """Folder with train/, test/, train.csv, test.csv: Task4/ if the images were copied there,
    else the competition folder next to the repository (same default as pis/)."""
    if (TASK_DIR / 'train.csv').is_file():
        return TASK_DIR
    return REPO_ROOT.parent / 'cpe342-kaggle-main' / '04_image_classification'


DATA_DIR = default_data_dir()  # replaced by --data-root in main()
EXP_ROOT = REPO_ROOT / 'outputs' / 'task4_exp'  # outputs/ is git-ignored
NUM_CLASSES = 5
CLASS_NAMES = ['7-11 Knights', 'FiveN', 'Free Fried', 'PUBE', 'Roblock']  # from datacard.txt
TEAM_LABELS = ['7-11k', 'fiveN', 'freefried', 'pube', 'roblock']  # names pis/submission.py maps back to 0-4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Task 4 experiments')
    parser.add_argument('--mode', choices=['train', 'leak-check', 'predict', 'ensemble', 'tta'], default='train',
                        help='train = train and grade a model; leak-check = look for near-duplicate '
                             'training images that fall on both sides of the split (round 1); '
                             'predict = label the test images with best.pt of --run-name; '
                             'ensemble = average the runs in --members and save the result as --run-name; '
                             'tta = logits of best.pt on its validation images and on test under every TTA view')
    parser.add_argument('--tta-views', default='',
                        help='comma list of views for --mode tta (default TTA_VIEWS); views already '
                             'saved for the same checkpoint are kept and only the new ones are computed')
    parser.add_argument('--members', default='',
                        help='comma-separated finished runs for --mode ensemble')
    parser.add_argument('--run-name', default='r0_baseline')
    parser.add_argument('--data-root', default='',
                        help='folder with train/, test/, train.csv, test.csv '
                             '(default: Task4/ if it holds train.csv, else ../cpe342-kaggle-main/04_image_classification)')
    parser.add_argument('--seed', type=int, default=342)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--lr-backbone', type=float, default=3e-5)
    parser.add_argument('--lr-head', type=float, default=3e-4)
    parser.add_argument('--probe-epochs', type=int, default=1,
                        help='first N epochs train the new head only (backbone frozen); 1 = previous runs. '
                             'Above 1 = LP-FT: the cosine schedule then starts when the backbone is unfrozen')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--jitter', type=float, default=0.1,
                        help='brightness/contrast/saturation jitter strength')
    parser.add_argument('--limit', type=int, default=0,
                        help='use only the first N training images (smoke test)')
    # Round 2 options. Defaults reproduce round 0.
    parser.add_argument('--split', choices=['hash', 'near-copy', 'scene'], default='hash',
                        help='hash = group byte-identical files (round 0); near-copy = also keep '
                             'near-copies of the same frame on one side (round 1 follow-up); '
                             'scene = hold out whole groups of look-alike screens (harder exam, round 5)')
    parser.add_argument('--aug', choices=['basic', 'strong', 'extreme', 'robust'], default='basic',
                        help='basic = colour jitter only; strong = rotation, hue, dark/bright, noise, blur; '
                             'extreme = strong, with half of the images damaged over much wider ranges; '
                             'robust = strong plus generic damage not tied to these images: crop/shift/stretch, '
                             'low resolution, JPEG compression, grey, a covered patch (task4-boost)')
    parser.add_argument('--schedule', choices=['constant', 'cosine'], default='constant',
                        help='cosine = lower the learning rate smoothly towards the end')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none',
                        help='balanced = mistakes on games with fewer images count more')
    # Round 3 option. Default reproduces rounds 0-2.
    parser.add_argument('--model', choices=CANDIDATE_MODELS, default='resnet18',
                        help='backbone (torchvision, ImageNet-1k weights only)')
    # Overnight options. Defaults reproduce rounds 0-3.
    parser.add_argument('--fold', type=int, choices=range(5), default=0,
                        help='which of the 5 parts is held out for grading (0 = the one used so far)')
    parser.add_argument('--micro-batch', type=int, default=0,
                        help='feed each batch in pieces of this size to save GPU memory (0 = whole batch). '
                             'Same result as the whole batch for models without BatchNorm (ConvNeXt)')
    parser.add_argument('--resume', action='store_true',
                        help='continue a stopped run from its last finished epoch (needs last.pt in the run '
                             'folder and the same settings); starts from epoch 1 if last.pt is missing')
    # task4-boost options. Defaults reproduce the runs above.
    parser.add_argument('--mix', choices=['none', 'cutmix', 'mixup', 'both'], default='none',
                        help='blend pairs of training images and their labels (CutMix pastes a patch, '
                             'Mixup overlays whole images); both = pick one at random per batch')
    parser.add_argument('--mix-prob', type=float, default=0.5, help='share of batches that are blended')
    parser.add_argument('--multi-scale', action='store_true',
                        help='each batch at a random size from 0.75x to 1.25x of --height/--width '
                             '(validation stays at --height x --width)')
    # Machine options: change how much of the PC the run uses, never the result.
    parser.add_argument('--cpu-threads', type=int, default=0,
                        help='limit PyTorch to this many CPU threads (0 = PyTorch default, all cores)')
    parser.add_argument('--low-priority', action='store_true',
                        help='ask Windows to run this training below normal priority so the PC stays usable')
    return parser.parse_args()


def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temporary file first, so stopping mid-save never leaves a broken last.pt."""
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


# Backbones that fit the 6 GB laptop GPU at 128x256, batch 128 (measured 2026-11-15).
# Left out: efficientnet_v2_s and swin_t (over 6 GB, 30-45 min per epoch), efficientnet_b2 (10 min per epoch).
# task4-boost (8 GB RTX 4070) adds larger torchvision backbones; all still ImageNet-1k weights only.
CANDIDATE_MODELS = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'regnet_y_1_6gf', 'convnext_tiny',
                    'convnext_small', 'convnext_base', 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_b3',
                    'swin_t', 'swin_s', 'swin_v2_t', 'regnet_y_3_2gf', 'regnet_y_8gf']


def build_model(model_name: str) -> tuple[nn.Module, str]:
    """ImageNet-1k backbone with its last layer replaced by a 5-game layer.

    Returns the model and the name prefix of the new last layer (fc. for ResNet/RegNet,
    classifier.1. for EfficientNet, classifier.2. for ConvNeXt), so training can give
    that layer its own learning rate and train it alone in epoch 1.
    """
    model = models.get_model(model_name, weights=imagenet_weights(model_name))
    head_name = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)][-1]
    parent_name, _, attribute = head_name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    old_head = getattr(parent, attribute)
    setattr(parent, attribute, nn.Linear(old_head.in_features, NUM_CLASSES))
    return model, head_name + '.'


ALLOWED_WEIGHT_NAMES = {'IMAGENET1K_V1', 'IMAGENET1K_V2'}  # ImageNet-1k only; never SWAG or other data


def imagenet_weights(model_name: str):
    """Pretrained weights that were trained on ImageNet-1k only (team rule for Task 4).

    Names the weight set explicitly instead of DEFAULT, whose meaning can change between
    torchvision versions, and refuses anything else (e.g. IMAGENET1K_SWAG_*, which was
    first trained on Instagram images).
    """
    available = [w for w in models.get_model_weights(model_name) if w.name in ALLOWED_WEIGHT_NAMES]
    if not available:
        raise SystemExit(f'{model_name} has no ImageNet-1k-only weights in torchvision; not allowed')
    return available[-1]  # V2 when it exists (a newer ImageNet-1k recipe), else V1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_images(train_frame: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    """Return all training images as one uint8 tensor (N, 3, H, W) plus SHA256 of each file.

    Decoding 31k JPEGs once and keeping them in RAM (about 2.6 GB) removes the
    per-epoch disk and decode cost. The result is cached under outputs/.
    """
    cache_dir = EXP_ROOT / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    pixel_cache, hash_cache = cache_dir / 'train_u8.npy', cache_dir / 'train_sha256.json'
    file_names = train_frame['file_name'].tolist()
    if pixel_cache.is_file() and hash_cache.is_file():
        cached = json.loads(hash_cache.read_text(encoding='utf-8'))
        if cached['file_names'] == file_names:
            return torch.from_numpy(np.load(pixel_cache)), cached['sha256']
    pixels: np.ndarray | None = None
    sha256: list[str] = []
    for index, file_name in enumerate(file_names):
        raw = (DATA_DIR / 'train' / file_name).read_bytes()
        sha256.append(hashlib.sha256(raw).hexdigest())
        with Image.open(DATA_DIR / 'train' / file_name) as image:
            array = np.asarray(image.convert('RGB'))
        if pixels is None:
            pixels = np.empty((len(file_names), *array.shape), dtype=np.uint8)
        if array.shape != pixels.shape[1:]:
            raise ValueError(f'{file_name} has size {array.shape}, expected {pixels.shape[1:]}')
        pixels[index] = array
        if (index + 1) % 5000 == 0:
            print(f'  decoded {index + 1}/{len(file_names)} images', flush=True)
    assert pixels is not None
    pixels = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2))  # NHWC -> NCHW
    np.save(pixel_cache, pixels)
    hash_cache.write_text(json.dumps({'file_names': file_names, 'sha256': sha256}), encoding='utf-8')
    return torch.from_numpy(pixels), sha256


def color_jitter(batch: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-image brightness, contrast and saturation jitter on a float batch in [0, 1]."""
    if strength <= 0:
        return batch
    def factor() -> torch.Tensor:
        return 1 + (torch.rand(batch.shape[0], 1, 1, 1, device=batch.device) * 2 - 1) * strength
    batch = (batch * factor()).clamp(0, 1)
    gray = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
    mean = gray.mean(dim=(1, 2, 3), keepdim=True)
    batch = ((batch - mean) * factor() + mean).clamp(0, 1)
    gray = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
    return ((batch - gray) * factor() + gray).clamp(0, 1)


RGB_TO_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def strong_augment(batch: torch.Tensor, extreme_share: float = 0.0, geometry: bool = True) -> torch.Tensor:
    """Damage training images the way the provided training images are already damaged.

    Training screenshots show heavy rotation with black corners, whole-image colour
    shifts, very dark or washed-out frames, grain noise and blur. Each image gets
    its own random amount of each. Order: colour -> blur -> noise -> rotation, so
    the black corners added by rotation stay clean black as in the data.

    extreme_share: fraction of images damaged with much wider ranges (--aug extreme).
    The validation stress test showed the model breaks on the far end of these same
    damages and on several at once; those images teach it that far end.
    """
    size, device = batch.shape[0], batch.device
    # No random draw when the share is 0, so --aug strong keeps its exact random sequence.
    extreme = (torch.rand(size, 1, 1, 1, device=device) < extreme_share if extreme_share > 0
               else torch.zeros(size, 1, 1, 1, dtype=torch.bool, device=device))
    extreme_flat = extreme.view(size)

    def uniform(low: float, high: float, wide_low: float | None = None, wide_high: float | None = None) -> torch.Tensor:
        normal = torch.empty(size, 1, 1, 1, device=device).uniform_(low, high)
        if wide_low is None or wide_high is None:
            return normal
        return torch.where(extreme, torch.empty(size, 1, 1, 1, device=device).uniform_(wide_low, wide_high), normal)

    def gray_of(x: torch.Tensor) -> torch.Tensor:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    batch = batch * uniform(0.4, 1.4, 0.15, 2.2)                         # dark .. bright
    mean = gray_of(batch).mean(dim=(1, 2, 3), keepdim=True)
    batch = (batch - mean) * uniform(0.6, 1.5, 0.3, 2.5) + mean          # contrast
    gray = gray_of(batch)
    batch = ((batch - gray) * uniform(0.3, 1.7, 0.0, 3.0) + gray).clamp(0, 1)   # saturation
    # Hue: rotate the two colour axes of YIQ space by a random angle (luminance unchanged).
    to_yiq = RGB_TO_YIQ.to(device)
    angle = torch.empty(size, device=device).uniform_(-np.pi, np.pi)
    angle = torch.where(torch.rand(size, device=device) < 0.8, angle, torch.zeros_like(angle))
    rotate = torch.zeros(size, 3, 3, device=device)
    rotate[:, 0, 0] = 1
    rotate[:, 1, 1], rotate[:, 1, 2] = torch.cos(angle), -torch.sin(angle)
    rotate[:, 2, 1], rotate[:, 2, 2] = torch.sin(angle), torch.cos(angle)
    transform = torch.linalg.inv(to_yiq) @ rotate @ to_yiq
    batch = torch.einsum('bij,bjhw->bihw', transform, batch).clamp(0, 1)
    # Blur on about 30% of images.
    blur = torch.rand(size, device=device) < 0.3
    if blur.any():
        kernel_1d = torch.exp(-torch.arange(-2, 3, device=device, dtype=batch.dtype) ** 2 / 2)
        kernel = (kernel_1d[:, None] * kernel_1d[None, :])
        kernel = (kernel / kernel.sum()).expand(3, 1, 5, 5)
        batch[blur] = F.conv2d(F.pad(batch[blur], (2, 2, 2, 2), mode='replicate'), kernel, groups=3)
    if extreme_share > 0:
        # Heavier blur (sigma 2.5) on about 40% of the extreme images.
        heavy = extreme_flat & (torch.rand(size, device=device) < 0.4)
        if heavy.any():
            kernel_1d = torch.exp(-torch.arange(-7, 8, device=device, dtype=batch.dtype) ** 2 / (2 * 2.5 ** 2))
            kernel = (kernel_1d[:, None] * kernel_1d[None, :])
            kernel = (kernel / kernel.sum()).expand(3, 1, 15, 15)
            batch[heavy] = F.conv2d(F.pad(batch[heavy], (7, 7, 7, 7), mode='replicate'), kernel, groups=3)
    batch = (batch + torch.randn_like(batch) * uniform(0.0, 0.10, 0.0, 0.30)).clamp(0, 1)   # grain noise
    if not geometry:
        return batch
    # Rotation (+-30 degrees) and zoom on about 70% of images; outside area becomes black.
    height, width = batch.shape[2], batch.shape[3]
    turn = torch.empty(size, device=device).uniform_(-np.pi / 6, np.pi / 6)
    turn = torch.where(torch.rand(size, device=device) < 0.7, turn, torch.zeros_like(turn))
    zoom = torch.empty(size, device=device).uniform_(0.9, 1.15)
    if extreme_share > 0:
        # Extreme images: rotation up to +-60 degrees (always) and zoom 0.7 .. 1.8.
        turn = torch.where(extreme_flat, torch.empty(size, device=device).uniform_(-np.pi / 3, np.pi / 3), turn)
        zoom = torch.where(extreme_flat, torch.empty(size, device=device).uniform_(0.7, 1.8), zoom)
    theta = torch.zeros(size, 2, 3, device=device)
    theta[:, 0, 0] = torch.cos(turn) / zoom
    theta[:, 0, 1] = -torch.sin(turn) * height / width / zoom
    theta[:, 1, 0] = torch.sin(turn) * width / height / zoom
    theta[:, 1, 1] = torch.cos(turn) / zoom
    grid = F.affine_grid(theta, list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def low_resolution(batch: torch.Tensor, share: float) -> torch.Tensor:
    """Shrink a share of the images to 30-90% of their size and blow them back up (soft, blocky detail)."""
    size, height, width = batch.shape[0], batch.shape[2], batch.shape[3]
    picked = torch.nonzero(torch.rand(size, device=batch.device) < share).flatten()
    for index in picked.tolist():
        factor = float(np.random.uniform(0.3, 0.9))
        small = F.interpolate(batch[index:index + 1], size=(max(8, int(height * factor)), max(8, int(width * factor))),
                              mode='area')
        if np.random.rand() < 0.3:
            batch[index:index + 1] = F.interpolate(small, size=(height, width), mode='nearest')
        else:
            batch[index:index + 1] = F.interpolate(small, size=(height, width), mode='bilinear', align_corners=False)
    return batch


def jpeg_compress(batch: torch.Tensor, share: float) -> torch.Tensor:
    """Save a share of the images as JPEG at quality 15-90 and decode them again (block artefacts)."""
    from torchvision.io import decode_jpeg, encode_jpeg
    picked = torch.nonzero(torch.rand(batch.shape[0], device=batch.device) < share).flatten()
    if len(picked) == 0:
        return batch
    as_bytes = (batch[picked] * 255).round().clamp(0, 255).to(torch.uint8)
    quality = int(np.random.randint(15, 91))  # one quality per batch keeps it to one encode call
    encoded = [e.cpu() for e in encode_jpeg(list(as_bytes), quality=quality)]
    decoded = decode_jpeg(encoded, device=batch.device)
    batch[picked] = torch.stack(decoded).float().div(255)
    return batch


def robust_augment(batch: torch.Tensor) -> torch.Tensor:
    """--aug robust: generic damage on top of strong_augment's colour/blur/noise.

    Not modelled on any image set: crops, shifts and stretches of the frame, low resolution,
    JPEG compression, grey frames and a covered patch are common ways screenshots of the same
    game can differ (other capture tools, window sizes, overlays).
    """
    size, device = batch.shape[0], batch.device
    batch = strong_augment(batch, extreme_share=0.0, geometry=False)
    batch = low_resolution(batch, share=0.3)
    batch = jpeg_compress(batch, share=0.4)
    gray = torch.rand(size, 1, 1, 1, device=device) < 0.05
    batch = torch.where(gray, (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]).expand_as(batch),
                        batch)
    # Geometry: rotation as in strong, plus separate x/y zoom (crop in or shrink with black border) and shift.
    height, width = batch.shape[2], batch.shape[3]
    turn = torch.empty(size, device=device).uniform_(-np.pi / 6, np.pi / 6)
    turn = torch.where(torch.rand(size, device=device) < 0.7, turn, torch.zeros_like(turn))
    zoom_x = torch.empty(size, device=device).uniform_(0.8, 1.6)
    zoom_y = zoom_x * torch.empty(size, device=device).uniform_(0.8, 1.25)
    shift = torch.empty(size, 2, device=device).uniform_(-0.2, 0.2)
    shift = torch.where(torch.rand(size, 1, device=device) < 0.6, shift, torch.zeros_like(shift))
    theta = torch.zeros(size, 2, 3, device=device)
    theta[:, 0, 0] = torch.cos(turn) / zoom_x
    theta[:, 0, 1] = -torch.sin(turn) * height / width / zoom_x
    theta[:, 1, 0] = torch.sin(turn) * width / height / zoom_y
    theta[:, 1, 1] = torch.cos(turn) / zoom_y
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(batch.shape), align_corners=False)
    batch = F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    # Covered patch (cutout) on 30% of images, filled with a random grey.
    covered = torch.nonzero(torch.rand(size, device=device) < 0.3).flatten().tolist()
    for index in covered:
        ph, pw = int(height * np.random.uniform(0.1, 0.35)), int(width * np.random.uniform(0.1, 0.35))
        top, left = np.random.randint(0, height - ph + 1), np.random.randint(0, width - pw + 1)
        batch[index, :, top:top + ph, left:left + pw] = float(np.random.rand())
    return batch


class Preprocess:
    """uint8 CPU batch -> resized, (optionally augmented), normalized float GPU batch."""

    def __init__(self, device: torch.device, height: int, width: int, jitter: float, aug: str = 'basic') -> None:
        self.device, self.size, self.jitter, self.aug = device, (height, width), jitter, aug
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, batch_u8: torch.Tensor, train: bool, view: str = 'id',
                 size: tuple[int, int] | None = None) -> torch.Tensor:
        batch = batch_u8.to(self.device, non_blocking=True).float().div_(255)
        if view.startswith('size'):  # TTA: whole image at another input size (model sees more/less detail)
            factor = float(view[4:])
            size = (int(round(self.size[0] * factor / 16)) * 16, int(round(self.size[1] * factor / 16)) * 16)
        batch = F.interpolate(batch, size=size or self.size, mode='bilinear', align_corners=False)
        if view != 'id' and not view.startswith('size'):
            batch = apply_view(batch, view)
        if train:
            if self.aug == 'robust':
                batch = robust_augment(batch)
            elif self.aug == 'extreme':
                batch = strong_augment(batch, extreme_share=0.5)
            elif self.aug == 'strong':
                batch = strong_augment(batch)
            else:
                batch = color_jitter(batch, self.jitter)
        batch = (batch - self.mean) / self.std
        return batch.contiguous(memory_format=torch.channels_last)


def mix_batch(inputs: torch.Tensor, kind: str, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """CutMix or Mixup inside one piece: returns blended inputs, partner order and the share of the own label."""
    order = torch.randperm(inputs.shape[0], device=inputs.device)
    lam = float(np.random.beta(alpha, alpha))
    if kind == 'mixup':
        return inputs * lam + inputs[order] * (1 - lam), order, lam
    height, width = inputs.shape[2], inputs.shape[3]
    cut_h, cut_w = int(height * np.sqrt(1 - lam)), int(width * np.sqrt(1 - lam))
    cy, cx = np.random.randint(height), np.random.randint(width)
    top, bottom = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    left, right = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    mixed = inputs.clone()
    mixed[:, :, top:bottom, left:right] = inputs[order, :, top:bottom, left:right]
    lam = 1 - (bottom - top) * (right - left) / (height * width)
    return mixed, order, lam


def predict_logits(model: nn.Module, images: torch.Tensor, indices: np.ndarray,
                   preprocess: Preprocess, batch_size: int, view: str = 'id') -> np.ndarray:
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size * 2):
            chunk = torch.from_numpy(indices[start:start + batch_size * 2])
            with torch.autocast(device_type='cuda'):
                outputs.append(model(preprocess(images[chunk], train=False, view=view)).float().cpu())
    return torch.cat(outputs).numpy()


# Test-time augmentation (TA ruling 15 Nov: allowed). Each view is a fixed, label-preserving
# change of the image; the softmax over several views is averaged. Which views to average
# is chosen on out-of-fold training predictions only (Task4/oof.py).
KNOWN_VIEWS = ['id', 'hflip', 'rot+8', 'rot-8', 'rot+16', 'rot-16', 'zoom1.12', 'zoom0.9',
               'size0.875', 'size1.125', 'size1.25', 'auto']
# Views computed by --mode tta. Chosen on scene fold 0 (training images only): +-8 degree turns and
# zoom in/out helped every model tried; flip hurt (screen text is mirrored) and size views were mixed.
TTA_VIEWS = ['id', 'rot+8', 'rot-8', 'zoom1.12', 'zoom0.9']


def affine(batch: torch.Tensor, turn: float, zoom: float) -> torch.Tensor:
    """Rotate by turn radians and zoom (>1 = closer); outside area black, as in strong_augment."""
    size, height, width = batch.shape[0], batch.shape[2], batch.shape[3]
    cos, sin = float(np.cos(turn)), float(np.sin(turn))
    theta = torch.tensor([[cos / zoom, -sin * height / width / zoom, 0.0],
                          [sin * width / height / zoom, cos / zoom, 0.0]], device=batch.device, dtype=batch.dtype)
    grid = F.affine_grid(theta.expand(size, 2, 3), list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def autocontrast(batch: torch.Tensor) -> torch.Tensor:
    """Per image: stretch its own 1st..99th brightness percentile to the full range, gain at most 4.

    Uses only the pixels of that one image. Aimed at dark and low-contrast frames, which hold most
    of the out-of-fold mistakes (training images only); well-exposed frames barely change.
    """
    gray = (0.299 * batch[:, 0] + 0.587 * batch[:, 1] + 0.114 * batch[:, 2]).flatten(1).float()
    low = torch.quantile(gray, 0.01, dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(gray, 0.99, dim=1).view(-1, 1, 1, 1)
    return ((batch - low) / (high - low).clamp(min=0.25)).clamp(0, 1)


def apply_view(batch: torch.Tensor, view: str) -> torch.Tensor:
    if view == 'auto':
        return autocontrast(batch)
    if view == 'hflip':
        return batch.flip(3)
    if view.startswith('rot'):
        return affine(batch, float(view[3:]) * np.pi / 180, 1.0)
    if view.startswith('zoom'):
        return affine(batch, 0.0, float(view[4:]))
    raise ValueError(f'unknown TTA view {view}')


def save_confusion_plot(matrix: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap='Blues')
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha='center', va='center',
                      color='white' if matrix[row, col] > matrix.max() / 2 else 'black')
    axis.set_xlabel('predicted class')
    axis.set_ylabel('actual class')
    axis.set_title('Task 4 validation confusion matrix')
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


SIMILARITY_BINS = [(0.0, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01)]


def frozen_features(images: torch.Tensor, preprocess: Preprocess, batch_size: int) -> torch.Tensor:
    """512-number description of every image from ResNet18 as trained on ImageNet (no training here).

    Two images that look alike get similar descriptions. Rows are scaled to length 1,
    so a dot product between two rows is their cosine similarity (1.0 = identical look).
    """
    model = models.resnet18(weights=imagenet_weights('resnet18'))
    model.fc = nn.Identity()
    model.to(preprocess.device, memory_format=torch.channels_last)
    features = predict_logits(model, images, np.arange(len(images)), preprocess, batch_size)
    features = torch.from_numpy(features).to(preprocess.device)
    return F.normalize(features, dim=1)


def nearest_neighbour(queries: torch.Tensor, pool: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, the most similar row in pool: (similarity, index)."""
    best_sim, best_idx = [], []
    for start in range(0, len(queries), 2048):
        similarity = queries[start:start + 2048] @ pool.T
        value, index = similarity.max(dim=1)
        best_sim.append(value.cpu())
        best_idx.append(index.cpu())
    return torch.cat(best_sim).numpy(), torch.cat(best_idx).numpy()


def save_pair_sheet(images: torch.Tensor, labels: np.ndarray, rows: list[tuple[str, list[tuple[int, int, float]]]],
                    path: Path) -> None:
    """Each row: one similarity range; each cell: validation image | its closest learning image."""
    from PIL import ImageDraw
    height, width = images.shape[2], images.shape[3]
    pairs_per_row = max(len(pairs) for _, pairs in rows)
    cell_width, label_height = width * 2 + 10, 16
    sheet = Image.new('RGB', (150 + pairs_per_row * cell_width, len(rows) * (height + label_height)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, (title, pairs) in enumerate(rows):
        top = row * (height + label_height)
        draw.text((5, top + height // 2), title, fill='black')
        for col, (val_i, fit_i, similarity) in enumerate(pairs):
            left = 150 + col * cell_width
            for offset, index in ((0, val_i), (width, fit_i)):
                sheet.paste(Image.fromarray(images[index].permute(1, 2, 0).numpy()), (left + offset, top))
            same = 'same game' if labels[val_i] == labels[fit_i] else 'DIFFERENT game'
            draw.text((left, top + height + 2), f'sim {similarity:.3f}  {same}', fill='black')
    sheet.save(path, quality=90)


def leak_check(images: torch.Tensor, labels: np.ndarray, sha256: list[str], fit_idx: np.ndarray,
               val_idx: np.ndarray, preprocess: Preprocess, run_dir: Path, batch_size: int) -> None:
    """Round 1: do near-copies of validation images sit in the learning part?

    Uses training images only. If many validation images have an almost identical
    image in the learning part, the validation score is inflated (the model has
    effectively seen the exam), and the split must group those copies together.
    """
    print('[3/4] Describe every training image with ImageNet ResNet18 (no training) '
          'and find each validation image\'s closest learning image', flush=True)
    start = time.perf_counter()
    features = frozen_features(images, preprocess, batch_size)
    np.save(EXP_ROOT / 'cache' / 'resnet18_imagenet_features.npy', features.cpu().numpy().astype(np.float16))
    sim_val, nn_local = nearest_neighbour(features[torch.from_numpy(val_idx)], features[torch.from_numpy(fit_idx)])
    nn_fit = fit_idx[nn_local]
    same_label = labels[val_idx] == labels[nn_fit]
    print(f'      done in {time.perf_counter() - start:.0f}s', flush=True)

    counts = pd.Series(sha256).value_counts()
    exact_copies = int(counts[counts > 1].sum())
    rng = np.random.default_rng(342)
    table, sheet_rows = [], []
    for low, high in SIMILARITY_BINS:
        in_bin = np.flatnonzero((sim_val >= low) & (sim_val < high))
        agree = float(same_label[in_bin].mean()) if len(in_bin) else float('nan')
        table.append({'similarity_range': f'{low:.2f}-{min(high, 1.0):.2f}', 'validation_images': int(len(in_bin)),
                      'share_of_validation': round(len(in_bin) / len(val_idx), 4),
                      'closest_image_same_game': round(agree, 4)})
        picks = rng.choice(in_bin, size=min(4, len(in_bin)), replace=False) if len(in_bin) else []
        sheet_rows.append((f'sim {low:.2f}-{min(high, 1.0):.2f}',
                           [(int(val_idx[i]), int(nn_fit[i]), float(sim_val[i])) for i in picks]))
    save_pair_sheet(images, labels, sheet_rows, run_dir / 'closest_pairs.jpg')
    pd.DataFrame(table).to_csv(run_dir / 'similarity_table.csv', index=False)
    pd.DataFrame({'val_index': val_idx, 'closest_fit_index': nn_fit, 'similarity': sim_val,
                  'same_game': same_label}).to_csv(run_dir / 'closest_pairs.csv', index=False)

    lines = ['[4/4] Result (round 1: leak check, training images only)',
             f'      byte-identical files in train: {exact_copies} images '
             f'(already kept on one side by the SHA256 grouping)',
             f'      overall: the closest learning image is the same game for '
             f'{same_label.mean():.1%} of validation images', '',
             '      similarity of each validation image to its closest learning image (1.00 = looks identical)',
             '        range       images  share   closest image is same game']
    for row in table:
        lines.append(f'        {row["similarity_range"]}  {row["validation_images"]:>6}  '
                     f'{row["share_of_validation"]:>6.1%}  {row["closest_image_same_game"]:.1%}')
    lines += ['', '      how to read: if the high-similarity rows hold many images and are almost always the',
              '      same game, and closest_pairs.jpg shows the same frame in both halves, the split leaks.',
              f'      files: {run_dir}',
              '        summary.txt, similarity_table.csv, closest_pairs.csv, closest_pairs.jpg']
    summary = '\n'.join(lines) + '\n'
    (run_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


NEAR_COPY_SIMILARITY = 0.93  # pairs at or above this looked like the same frame in closest_pairs.jpg
NEAR_COPY_MIN_CONTRAST = 0.08  # near-black / washed-out frames look alike without being the same frame
NEAR_COPY_MAX_GROUP = 5  # larger clusters were chains of unrelated dark or noisy frames
SCENES_PER_GAME = 20  # look-alike groups per game for --split scene
# Options that never change how a model is trained; --resume ignores them when comparing settings.
NOT_TRAINING_SETTINGS = ('resume', 'cpu_threads', 'low_priority', 'members', 'data_root', 'tta_views')


def near_copy_groups(images: torch.Tensor, labels: np.ndarray, sha256: list[str], preprocess: Preprocess,
                     batch_size: int) -> tuple[list[str], dict]:
    """Group id per training image so that near-copies of one frame stay on one side of the split.

    Two images are linked when ImageNet ResNet18 sees them as very similar (cosine >= 0.93,
    checked among each image's 5 closest images) and neither is a near-blank frame.
    Linked images form groups; only small groups with one label are kept (checked by eye:
    those are the same frame recoloured or rotated). Everything else keeps its SHA256 group.
    Uses training images only.
    """
    cache = EXP_ROOT / 'cache' / 'resnet18_imagenet_features.npy'
    features = None
    if cache.is_file():
        cached = np.load(cache)
        if len(cached) == len(images):
            features = F.normalize(torch.from_numpy(cached).float().to(preprocess.device), dim=1)
    if features is None:
        features = frozen_features(images, preprocess, batch_size)
    contrast = torch.cat([images[start:start + 4096].to(preprocess.device).float().div(255).mean(1)
                          .flatten(1).std(1) for start in range(0, len(images), 4096)]).cpu().numpy()
    neighbour_sim, neighbour_idx = [], []
    for start in range(0, len(features), 2048):
        value, index = (features[start:start + 2048] @ features.T).topk(6, dim=1)
        neighbour_sim.append(value[:, 1:].cpu())
        neighbour_idx.append(index[:, 1:].cpu())
    neighbour_sim, neighbour_idx = torch.cat(neighbour_sim).numpy(), torch.cat(neighbour_idx).numpy()

    parent = np.arange(len(images))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    usable = contrast >= NEAR_COPY_MIN_CONTRAST
    for node in np.flatnonzero(usable):
        for similarity, other in zip(neighbour_sim[node], neighbour_idx[node]):
            if similarity >= NEAR_COPY_SIMILARITY and usable[other]:
                parent[find(node)] = find(other)
    roots = np.array([find(node) for node in range(len(images))])
    groups = list(sha256)
    kept_groups = kept_images = dropped_groups = 0
    for root, members in pd.Series(np.arange(len(images))).groupby(roots):
        members = members.to_numpy()
        if len(members) < 2:
            continue
        if len(members) <= NEAR_COPY_MAX_GROUP and len(set(labels[members])) == 1:
            for member in members:
                groups[member] = f'near-copy-{root}'
            kept_groups, kept_images = kept_groups + 1, kept_images + len(members)
        else:
            dropped_groups += 1
    stats = {'near_copy_groups': kept_groups, 'images_in_near_copy_groups': kept_images,
             'rejected_clusters': dropped_groups}
    return groups, stats


def scene_groups(images: torch.Tensor, labels: np.ndarray, preprocess: Preprocess, batch_size: int,
                 seed: int) -> tuple[list[str], dict]:
    """Group id per training image so that whole scenes are held out together.

    Images of each game are clustered into SCENES_PER_GAME look-alike groups (k-means on
    ImageNet ResNet18 features). A whole group lands on one side of the split, so the
    model is graded on kinds of screens it never saw while learning. This is a harder
    exam than near-copy, meant to imitate screenshots from unseen play sessions.
    Uses training images only.
    """
    from sklearn.cluster import KMeans

    cache = EXP_ROOT / 'cache' / 'resnet18_imagenet_features.npy'
    features = None
    if cache.is_file():
        cached = np.load(cache)
        if len(cached) == len(images):
            features = cached.astype(np.float32)
    if features is None:
        features = frozen_features(images, preprocess, batch_size).cpu().numpy().astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    groups = [''] * len(images)
    for game in range(NUM_CLASSES):
        members = np.flatnonzero(labels == game)
        scenes = KMeans(min(SCENES_PER_GAME, len(members)), n_init=3, random_state=seed).fit_predict(features[members])
        for member, scene in zip(members, scenes):
            groups[member] = f'scene-{game}-{scene}'
    return groups, {'scene_groups': len(set(groups))}


def load_test_images(test_frame: pd.DataFrame) -> torch.Tensor:
    """All test images as one uint8 tensor (N, 3, H, W), cached like the training images.

    Only used to label test images; nothing computed from them feeds back into training.
    """
    cache_dir = EXP_ROOT / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    pixel_cache, name_cache = cache_dir / 'test_u8.npy', cache_dir / 'test_files.json'
    file_names = test_frame['file_name'].tolist()
    if pixel_cache.is_file() and name_cache.is_file():
        if json.loads(name_cache.read_text(encoding='utf-8')) == file_names:
            return torch.from_numpy(np.load(pixel_cache))
    arrays = [np.asarray(Image.open(DATA_DIR / 'test' / name).convert('RGB')) for name in file_names]
    pixels = np.ascontiguousarray(np.stack(arrays).transpose(0, 3, 1, 2))
    np.save(pixel_cache, pixels)
    name_cache.write_text(json.dumps(file_names), encoding='utf-8')
    return torch.from_numpy(pixels)


def load_run_model(run_dir: Path, device: torch.device, which: str = 'best') -> tuple[nn.Module, dict, int]:
    checkpoint_path = run_dir / f'{which}.pt'
    if not checkpoint_path.is_file():
        raise SystemExit(f'{checkpoint_path} not found: train this run first')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    settings = checkpoint['args']
    model, _ = build_model(settings['model'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device, memory_format=torch.channels_last)
    model.eval()
    return model, settings, int(checkpoint['epoch'])


def tta_logits(run_dir: Path, device: torch.device, views: list[str] | None = None) -> None:
    """Logits under each view (default TTA_VIEWS) on the run's validation images and on test.

    Uses final.pt (last epoch, not chosen on validation) when it exists, else best.pt.
    Saves val_logits_tta.npy / test_logits_tta.npy, shape (views, images, 5), and tta_views.json.
    Views already saved for the same checkpoint are kept; only missing views are computed and appended.
    Picking the views to average is done later on validation (out-of-fold) predictions only.
    """
    views = views or TTA_VIEWS
    unknown = [v for v in views if v not in KNOWN_VIEWS]
    if unknown:
        raise SystemExit(f'unknown TTA views {unknown}; known: {KNOWN_VIEWS}')
    which = 'final' if (run_dir / 'final.pt').is_file() else 'best'
    done_views, old_val, old_test = [], None, None
    info_file = run_dir / 'tta_views.json'
    if (run_dir / 'val_logits_tta.npy').is_file() and info_file.is_file():
        info = json.loads(info_file.read_text(encoding='utf-8'))
        if isinstance(info, dict) and info.get('checkpoint') == which:
            done_views = info['views']
            old_val, old_test = np.load(run_dir / 'val_logits_tta.npy'), np.load(run_dir / 'test_logits_tta.npy')
    todo = [v for v in views if v not in done_views]
    if not todo:
        print(f'{run_dir.name}: views {views} already saved')
        return
    model, settings, epoch = load_run_model(run_dir, device, which)
    preprocess = Preprocess(device, settings['height'], settings['width'], 0.0, settings.get('aug', 'basic'))
    piece = settings.get('micro_batch') or settings['batch_size']
    train_frame = pd.read_csv(DATA_DIR / 'train.csv')
    images, _ = load_train_images(train_frame)
    labels = train_frame['label'].to_numpy(dtype=int)
    val_idx = np.load(run_dir / 'val_idx.npy')
    test_images = load_test_images(pd.read_csv(DATA_DIR / 'test.csv'))
    test_idx = np.arange(len(test_images))
    print(f'=== Task 4 TTA logits for "{run_dir.name}" ({settings["model"]}, {which}.pt = epoch {epoch}) ===',
          flush=True)
    val_views, test_views = [], []
    for view in todo:
        start = time.perf_counter()
        val_views.append(predict_logits(model, images, val_idx, preprocess, piece, view))
        test_views.append(predict_logits(model, test_images, test_idx, preprocess, piece, view))
        print(f'      {view:9s} validation macro F1 {macro_f1(labels[val_idx], val_views[-1].argmax(1)):.4f}'
              f'  ({time.perf_counter() - start:.0f}s)', flush=True)
    new_val, new_test = np.stack(val_views).astype(np.float32), np.stack(test_views).astype(np.float32)
    if old_val is not None:
        new_val, new_test = np.concatenate([old_val, new_val]), np.concatenate([old_test, new_test])
    np.save(run_dir / 'val_logits_tta.npy', new_val)
    np.save(run_dir / 'test_logits_tta.npy', new_test)
    info_file.write_text(json.dumps({'views': done_views + todo, 'checkpoint': which, 'epoch': epoch}),
                         encoding='utf-8')


def predict_test(run_dir: Path, device: torch.device) -> None:
    """Label every test image with best.pt of a finished run.

    Writes outputs/main/task4_predictions.csv (team format for `python run.py assemble`)
    and, when baseline/submission.csv exists, a ready-to-upload copy of it with only
    the task4 column replaced (matched on the ANS id in test.csv).
    """
    checkpoint_path = run_dir / 'best.pt'
    if not checkpoint_path.is_file():
        raise SystemExit(f'{checkpoint_path} not found: train this run first')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    settings = checkpoint['args']
    model, _ = build_model(settings['model'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device, memory_format=torch.channels_last)
    print(f'=== Task 4 predict with "{run_dir.name}" ({settings["model"]}, epoch {checkpoint["epoch"]}, '
          f'{settings["width"]}x{settings["height"]}) ===', flush=True)
    test = pd.read_csv(DATA_DIR / 'test.csv')
    preprocess = Preprocess(device, settings['height'], settings['width'], 0.0, settings.get('aug', 'basic'))
    piece = settings.get('micro_batch') or settings['batch_size']
    model.eval()
    outputs: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(test), piece):
            names = test['file_name'].iloc[begin:begin + piece]
            arrays = [np.asarray(Image.open(DATA_DIR / 'test' / name).convert('RGB')) for name in names]
            batch = torch.from_numpy(np.ascontiguousarray(np.stack(arrays).transpose(0, 3, 1, 2)))
            with torch.autocast(device_type='cuda'):
                outputs.append(model(preprocess(batch, train=False)).float().cpu().numpy())
    logits = np.concatenate(outputs)
    np.save(run_dir / 'test_logits.npy', logits)
    print(f'      {len(test):,} test images labelled in {time.perf_counter() - start:.0f}s', flush=True)
    write_test_outputs(run_dir, test, logits.argmax(1))


def ensemble_test(run_dir: Path, members: list[str], device: torch.device) -> None:
    """Average the softmax of several finished runs on test (soft voting, equal weights).

    Members are chosen before looking at any ensemble score. When all members were graded
    on the same validation images, the ensemble's validation score is printed too. Members
    trained on different splits are allowed (they add variety); then the ensemble is scored
    only on the images every member held out, a small sample. Test images are only
    labelled, never used to choose anything.
    """
    if len(members) < 2:
        raise SystemExit('--members needs at least two run names, e.g. --members r3d_lr2e-4,r3d_lr3e-4')
    member_dirs = [EXP_ROOT / name for name in members]
    labels_all = pd.read_csv(DATA_DIR / 'train.csv')['label'].to_numpy(dtype=int)
    val_sets = [np.load(member_dir / 'val_idx.npy') for member_dir in member_dirs]
    same_split = all(np.array_equal(val_set, val_sets[0]) for val_set in val_sets)
    val_probs, test_probs = [], []
    for member_dir, val_idx in zip(member_dirs, val_sets):
        member_val = torch.softmax(torch.from_numpy(np.load(member_dir / 'val_logits.npy')).float(), 1).numpy()
        val_probs.append(member_val)
        print(f'      {member_dir.name:18s} macro F1 on its own validation {macro_f1(labels_all[val_idx], member_val.argmax(1)):.4f}')
        if not (member_dir / 'test_logits.npy').is_file():
            predict_test(member_dir, device)
        test_probs.append(torch.softmax(torch.from_numpy(np.load(member_dir / 'test_logits.npy')).float(), 1).numpy())
    run_dir.mkdir(parents=True, exist_ok=True)
    if same_split:
        val_mean = np.mean(val_probs, axis=0)
        print(f'      ensemble ({len(members)} runs) validation macro F1 '
              f'{macro_f1(labels_all[val_sets[0]], val_mean.argmax(1)):.4f}')
        np.save(run_dir / 'val_probs.npy', val_mean)
    else:
        # Only images held out by every member are unseen by all of them; score the ensemble on those.
        common = np.array(sorted(set.intersection(*(set(val_set.tolist()) for val_set in val_sets))))
        if len(common):
            picked = [probs[[np.flatnonzero(val_set == image)[0] for image in common]]
                      for probs, val_set in zip(val_probs, val_sets)]
            print(f'      ensemble ({len(members)} runs) macro F1 on the {len(common):,} images no member learned '
                  f'{macro_f1(labels_all[common], np.mean(picked, axis=0).argmax(1)):.4f} (small sample: rough only)')
        else:
            print(f'      ensemble ({len(members)} runs): no image was held out by every member, '
                  f'so the combined score cannot be measured locally')
    test_mean = np.mean(test_probs, axis=0)
    np.save(run_dir / 'test_probs.npy', test_mean)
    (run_dir / 'members.txt').write_text('\n'.join(members) + '\n', encoding='utf-8')
    write_test_outputs(run_dir, pd.read_csv(DATA_DIR / 'test.csv'), test_mean.argmax(1))


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(truth, predicted, average='macro', zero_division=0))


def write_test_outputs(run_dir: Path, test: pd.DataFrame, predicted: np.ndarray) -> None:
    """Team prediction file plus a ready-to-upload copy of baseline/submission.csv (task4 replaced)."""
    stems = test['file_name'].str.replace(r'\.[A-Za-z0-9]+$', '', regex=True)
    team_dir = REPO_ROOT / 'outputs' / 'main'
    team_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'player_id': stems, 'prediction': [TEAM_LABELS[i] for i in predicted]}).to_csv(
        team_dir / 'task4_predictions.csv', index=False)
    print(f'      wrote {team_dir / "task4_predictions.csv"} (for python run.py assemble)')

    baseline_path = REPO_ROOT / 'baseline' / 'submission.csv'
    if baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        by_id = pd.Series(predicted, index=test['id'].astype(str))
        if not by_id.index.is_unique or set(by_id.index) != set(baseline['id'].astype(str)):
            raise SystemExit('test.csv ids do not match baseline/submission.csv ids')
        submission = baseline.copy()
        submission['task4'] = by_id.loc[baseline['id'].astype(str)].to_numpy().astype(int)
        submission_path = run_dir / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        reread = pd.read_csv(submission_path)
        assert list(reread.columns) == list(baseline.columns) and (reread['id'] == baseline['id']).all()
        assert all(reread[c].equals(baseline[c]) for c in baseline.columns if c not in ('id', 'task4'))
        assert set(reread['task4'].unique()) <= set(range(NUM_CLASSES)) and not reread.isna().any().any()
        print(f'      wrote {submission_path} (baseline/submission.csv with task4 replaced)')


def main() -> None:
    global DATA_DIR
    args = parse_args()
    if args.data_root:
        DATA_DIR = Path(args.data_root).resolve()
    if not (DATA_DIR / 'train.csv').is_file():
        raise SystemExit(f'{DATA_DIR} has no train.csv: pass --data-root <folder with train/, test/, train.csv>')
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    if args.low_priority:
        import ctypes
        if sys.platform == 'win32':  # BELOW_NORMAL_PRIORITY_CLASS; handles are 64-bit, so declare the types
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000):
                print('      warning: could not lower the process priority', flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available. Install a CUDA build of PyTorch (e.g. the .venv-gpu environment)')
    device = torch.device('cuda')
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = EXP_ROOT / args.run_name
    if args.mode == 'predict':
        predict_test(run_dir, device)
        return
    if args.mode == 'tta':
        tta_logits(run_dir, device, [v for v in args.tta_views.split(',') if v] or None)
        return
    if args.mode == 'ensemble':
        print(f'=== Task 4 ensemble "{args.run_name}" ===', flush=True)
        ensemble_test(run_dir, [name.strip() for name in args.members.split(',') if name.strip()], device)
        return
    results_file = EXP_ROOT / 'results.csv'
    if args.resume and results_file.is_file() and args.run_name in set(pd.read_csv(results_file)['run_name']):
        # Lets a whole queue be pasted again after a stop: finished runs are skipped.
        print(f'=== "{args.run_name}" already finished (in results.csv); skipped because of --resume ===')
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Task 4 experiment "{args.run_name}" on {torch.cuda.get_device_name(0)} ===')
    print('[1/4] Load training images into RAM (test images are never opened)', flush=True)
    train_frame = pd.read_csv(DATA_DIR / 'train.csv')
    load_start = time.perf_counter()
    images, sha256 = load_train_images(train_frame)
    print(f'      {len(images):,} images of {images.shape[3]}x{images.shape[2]} px '
          f'loaded in {time.perf_counter() - load_start:.0f}s', flush=True)
    labels = train_frame['label'].to_numpy(dtype=int).copy()
    if args.limit:
        images, sha256, labels = images[:args.limit], sha256[:args.limit], labels[:args.limit]

    print('[2/4] Split training images: the model learns from one part and is graded on the other')
    preprocess = Preprocess(device, args.height, args.width, args.jitter, args.aug)
    # hash: same holdout as pis/image.py (byte-identical images stay on one side).
    groups, split_stats = list(sha256), {}
    # The near-copy and scene groups come from GPU features, which can differ in the last digits
    # between runs; the groups are cached so every run of one split holds out exactly the same images.
    groups_cache = EXP_ROOT / 'cache' / f'groups_{args.split}_seed{args.seed}_n{len(labels)}.json'
    if args.split != 'hash' and groups_cache.is_file():
        cached = json.loads(groups_cache.read_text(encoding='utf-8'))
        groups, split_stats = cached['groups'], cached['stats']
        print(f'      groups for split "{args.split}" read from {groups_cache.name}')
    elif args.split == 'near-copy':
        groups, split_stats = near_copy_groups(images, labels, sha256, preprocess, args.batch_size)
        print(f'      near-copies kept together: {split_stats["near_copy_groups"]} groups, '
              f'{split_stats["images_in_near_copy_groups"]} images '
              f'({split_stats["rejected_clusters"]} unreliable clusters ignored)')
    elif args.split == 'scene':
        groups, split_stats = scene_groups(images, labels, preprocess, args.batch_size, args.seed)
        print(f'      whole scenes kept together: {split_stats["scene_groups"]} look-alike groups '
              f'({SCENES_PER_GAME} per game)')
    if args.split != 'hash' and not groups_cache.is_file():
        groups_cache.write_text(json.dumps({'groups': groups, 'stats': split_stats}), encoding='utf-8')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fit_idx, val_idx = list(splitter.split(np.zeros(len(labels)), labels, groups=groups))[args.fold]
    print(f'      fold {args.fold}: learn on {len(fit_idx):,} images, grade on {len(val_idx):,} images (validation)',
          flush=True)
    piece = args.micro_batch or args.batch_size  # images sent to the GPU at once

    if args.mode == 'leak-check':
        leak_check(images, labels, sha256, fit_idx, val_idx, preprocess, run_dir, args.batch_size)
        return
    model, head_prefix = build_model(args.model)
    model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith(head_prefix)], 'lr': args.lr_backbone},
        {'params': [p for n, p in model.named_parameters() if n.startswith(head_prefix)], 'lr': args.lr_head}],
        weight_decay=args.weight_decay)
    class_weight = None
    if args.class_weight == 'balanced':
        counts = np.bincount(labels[fit_idx], minlength=NUM_CLASSES)
        class_weight = torch.tensor(len(fit_idx) / (NUM_CLASSES * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    steps_per_epoch = -(-len(fit_idx) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if not 1 <= args.probe_epochs < args.epochs:
        raise SystemExit(f'--probe-epochs must be at least 1 and below --epochs ({args.epochs})')
    scheduler = None
    if args.schedule == 'cosine' and args.probe_epochs == 1:
        # Smoothly lower both learning rates to 1% of their start by the last step.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(step, total_steps) / total_steps)))
    elif args.schedule == 'cosine':
        # LP-FT: full learning rate while only the head learns, then the same cosine
        # as above over the fine-tuning steps alone, so fine-tuning matches a normal run.
        probe_steps = args.probe_epochs * steps_per_epoch
        tune_steps = total_steps - probe_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 if step < probe_steps else
            0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(step - probe_steps, tune_steps) / tune_steps)))
    scaler = torch.amp.GradScaler('cuda')
    label_tensor = torch.from_numpy(labels).long()  # int64 for the loss (numpy int is 32-bit on some Windows builds)
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    print(f'[3/4] Train {args.model} (ImageNet-pretrained) for {args.epochs} epochs at {args.width}x{args.height} px')
    print(f'      settings: split={args.split}  aug={args.aug}  schedule={args.schedule}  '
          f'label_smoothing={args.label_smoothing}  class_weight={args.class_weight}  '
          f'head-only epochs={args.probe_epochs}')
    print('      one epoch = the model sees every training image once; after each epoch it is graded')
    print('      loss = training error (lower is better); macro F1 = grade on validation, 0 to 1 (higher is better)',
          flush=True)
    history: list[dict] = []
    best_score, best_logits = -1.0, None
    first_epoch = 0
    last_path = run_dir / 'last.pt'
    run_settings = {k: v for k, v in vars(args).items() if k not in NOT_TRAINING_SETTINGS}
    if args.resume and last_path.is_file():
        # Load to CPU: random-number states must stay CPU tensors; load_state_dict moves weights to the GPU.
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state['settings'] != run_settings:
            changed = [k for k in run_settings if state['settings'].get(k) != run_settings[k]]
            raise SystemExit(f'--resume: settings differ from the stopped run ({", ".join(changed)}). '
                             'Use the same command as before, or drop --resume to start over.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        shuffle_generator.set_state(state['shuffle_generator'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        history, best_score, best_logits = state['history'], state['best_score'], state['best_logits']
        first_epoch = state['epoch']
        print(f'      resumed from {last_path.name}: epochs 1-{first_epoch} already done, '
              f'best macro F1 so far {best_score:.4f}', flush=True)
    elif args.resume:
        print('      --resume: no last.pt in the run folder, starting from epoch 1', flush=True)
    elif last_path.is_file():
        print('      note: an unfinished last.pt exists; starting over from epoch 1 (add --resume to continue it)',
              flush=True)
    for epoch in range(first_epoch, args.epochs):
        epoch_start = time.perf_counter()
        # The first --probe-epochs epochs train the new head only, with frozen BatchNorm
        # statistics (1 epoch, as in the baseline, unless --probe-epochs says more).
        probing = epoch < args.probe_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not probing or name.startswith(head_prefix)
        model.train()
        if probing:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        order = torch.from_numpy(fit_idx)[torch.randperm(len(fit_idx), generator=shuffle_generator)]
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            chunk = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            # One weight update per batch; the batch may be fed in pieces to fit in GPU memory.
            # Each piece's mean loss is weighted by its share of the batch, so the summed
            # gradient equals the gradient of the whole batch's mean loss.
            batch_size_now = None
            if args.multi_scale:
                scale = float(np.random.choice([0.75, 0.875, 1.0, 1.125, 1.25]))
                batch_size_now = (int(round(args.height * scale / 16)) * 16, int(round(args.width * scale / 16)) * 16)
            mix_kind = 'none'
            if args.mix != 'none' and np.random.rand() < args.mix_prob:
                mix_kind = args.mix if args.mix != 'both' else ('cutmix' if np.random.rand() < 0.5 else 'mixup')
            for piece_start in range(0, len(chunk), piece):
                part = chunk[piece_start:piece_start + piece]
                inputs = preprocess(images[part], train=True, size=batch_size_now)
                targets = label_tensor[part].to(device, non_blocking=True)
                if mix_kind != 'none':
                    inputs, targets_b, lam = mix_batch(inputs, mix_kind)
                    with torch.autocast(device_type='cuda'):
                        output = model(inputs)
                        loss = lam * criterion(output, targets) + (1 - lam) * criterion(output, targets[targets_b])
                else:
                    with torch.autocast(device_type='cuda'):
                        loss = criterion(model(inputs), targets)
                scaler.scale(loss * len(part) / len(chunk)).backward()
                loss_sum += loss.item() * len(part)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        logits = predict_logits(model, images, val_idx, preprocess, piece)
        score = float(f1_score(labels[val_idx], logits.argmax(1), average='macro', zero_division=0))
        seconds = time.perf_counter() - epoch_start
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(fit_idx),
                        'validation_macro_f1': score, 'seconds': round(seconds, 1)})
        is_best = score > best_score
        print(f'      epoch {epoch + 1}/{args.epochs}: loss={loss_sum / len(fit_idx):.4f}  '
              f'macro F1={score:.4f}  ({seconds:.0f}s){"  <- best so far, saved" if is_best else ""}',
              flush=True)
        if is_best:
            best_score, best_logits = score, logits
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch + 1, 'args': vars(args)},
                       run_dir / 'best.pt')
        if epoch + 1 == args.epochs:
            # The last epoch is kept too: choosing it needs no look at the validation score,
            # so its validation result is an honest estimate (best.pt is picked on validation).
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch + 1, 'args': vars(args)},
                       run_dir / 'final.pt')
            np.save(run_dir / 'val_logits_final.npy', logits)
        # Everything needed to continue from the next epoch after a stop (--resume).
        save_checkpoint(last_path, {
            'settings': run_settings, 'epoch': epoch + 1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'shuffle_generator': shuffle_generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(), 'history': history, 'best_score': best_score,
            'best_logits': best_logits})

    assert best_logits is not None
    predicted = best_logits.argmax(1)
    matrix = confusion_matrix(labels[val_idx], predicted, labels=list(range(NUM_CLASSES)))
    report = classification_report(labels[val_idx], predicted, labels=list(range(NUM_CLASSES)),
                                   output_dict=True, zero_division=0)
    np.save(run_dir / 'val_logits.npy', best_logits)
    np.save(run_dir / 'val_idx.npy', val_idx)
    save_confusion_plot(matrix, run_dir / 'confusion_matrix.png')
    total_seconds = sum(row['seconds'] for row in history)
    metrics = {'run_name': args.run_name, 'args': vars(args), 'device': torch.cuda.get_device_name(0),
               'train_size': int(len(fit_idx)), 'validation_size': int(len(val_idx)),
               'split': f'StratifiedGroupKFold(5) grouped by {args.split}, first fold held out',
               'split_stats': split_stats,
               'best_validation_macro_f1': best_score, 'train_seconds': total_seconds,
               'peak_gpu_memory_mb': int(torch.cuda.max_memory_allocated() / 2 ** 20),
               'classification_report': report, 'confusion_matrix': matrix.tolist(), 'history': history}
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    # One row per run so experiments can be compared in a single table.
    results_path = EXP_ROOT / 'results.csv'
    columns = ['run_name', 'model', 'split', 'fold', 'aug', 'schedule', 'label_smoothing', 'class_weight',
               'lr_backbone', 'lr_head', 'height', 'width', 'epochs', 'limit', 'mix', 'multi_scale', 'val_macro_f1', *[f'f1_class{c}' for c in range(NUM_CLASSES)],
               'train_minutes', 'peak_gpu_mb']
    row = {'run_name': args.run_name, 'model': args.model, 'split': args.split, 'fold': args.fold, 'aug': args.aug,
           'schedule': args.schedule, 'label_smoothing': args.label_smoothing, 'class_weight': args.class_weight,
           'lr_backbone': args.lr_backbone, 'lr_head': args.lr_head,
           'height': args.height, 'width': args.width, 'epochs': args.epochs, 'limit': args.limit,
           'mix': args.mix, 'multi_scale': args.multi_scale,
           'val_macro_f1': round(best_score, 4),
           **{f'f1_class{c}': round(report[str(c)]['f1-score'], 4) for c in range(NUM_CLASSES)},
           'train_minutes': round(total_seconds / 60, 1), 'peak_gpu_mb': metrics['peak_gpu_memory_mb']}
    previous = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame(columns=columns)
    # Older rows (round 0) lack the newer setting columns; they used the defaults.
    defaults = {'split': 'hash', 'aug': 'basic', 'schedule': 'constant', 'label_smoothing': 0.0, 'class_weight': 'none',
                'fold': 0, 'lr_backbone': 3e-5, 'lr_head': 3e-4, 'mix': 'none', 'multi_scale': False}
    previous = previous.assign(**{k: previous[k] if k in previous else v for k, v in defaults.items()})
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True)[columns].to_csv(results_path, index=False)
    summary = human_summary(args.run_name, best_score, int(saved_epoch(history, best_score)), total_seconds,
                            metrics['peak_gpu_memory_mb'], report, matrix, run_dir, history)
    (run_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    last_path.unlink(missing_ok=True)  # the run is complete; nothing left to resume
    print(summary)


def saved_epoch(history: list[dict], best_score: float) -> int:
    return next(row['epoch'] for row in history if row['validation_macro_f1'] == best_score)


def human_summary(run_name: str, best_score: float, best_epoch: int, total_seconds: float, peak_gpu_mb: int,
                  report: dict, matrix: np.ndarray, run_dir: Path, history: list[dict]) -> str:
    """Plain-language result of one run, printed and saved as summary.txt."""
    lines = ['[4/4] Result', f'      run: {run_name}',
             f'      overall grade (macro F1): {best_score:.4f}  from epoch {best_epoch}',
             f'      training time: {total_seconds / 60:.1f} min, GPU memory peak: {peak_gpu_mb} MB', '',
             '      per epoch: loss = training error (lower is better), macro F1 = validation grade',
             *[f'        epoch {row["epoch"]:>2}: loss={row["train_loss"]:.4f}  macro F1={row["validation_macro_f1"]:.4f}'
               f'  ({row["seconds"]:.0f}s)' for row in history], '',
             '      per game: how often the model is right on validation images']
    for c, name in enumerate(CLASS_NAMES):
        total, correct = int(matrix[c].sum()), int(matrix[c, c])
        wrong = matrix[c].copy()
        wrong[c] = 0
        confused = int(wrong.argmax())
        lines.append(f'        {c} {name:<13} F1={report[str(c)]["f1-score"]:.3f}  correct {correct}/{total}'
                     f'  most often mistaken for: {CLASS_NAMES[confused]} ({int(wrong[confused])} images)')
    weakest = min(range(NUM_CLASSES), key=lambda c: report[str(c)]['f1-score'])
    lines += ['', f'      weakest game: {CLASS_NAMES[weakest]}',
              f'      files: {run_dir}', '        summary.txt, metrics.json, confusion_matrix.png',
              f'      all runs compared in: {EXP_ROOT / "results.csv"}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
