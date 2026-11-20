"""Task 4: label any folder of game screenshots with trained models (no training needed).

Give it one or more trained checkpoints (best.pt files) and a folder of images.
With several checkpoints it averages their softmax with equal weights, the same
soft voting as `train.py --mode ensemble`. Each checkpoint stores its own input
size and architecture, so models trained at different sizes can be mixed.

    # pack the 18 models of ens18_5fold into one folder to hand to a teammate
    python Task4/predict.py --pack task4_models --members-file outputs/task4_exp/ens18_5fold/members.txt
    # label every image in a folder (all *.pt files in task4_models are used)
    python Task4/predict.py --models task4_models --images path/to/images --out task4_new.csv

Output csv: file_name, prediction (0-4, the index Kaggle scores), game (name),
then one probability column per game.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CLASS_NAMES, EXP_ROOT, NUM_CLASSES, TEAM_LABELS, Preprocess  # noqa: E402

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

# Test-time augmentation (--tta): each model labels every image in these fixed views
# and the softmax is averaged. All views stay inside the ranges --aug strong trained
# on (rotation +-30 degrees, zoom 0.9-1.15, brightness 0.4-1.4), so the model has seen
# images like them. (name, rotation in degrees, zoom, brightness factor)
TTA_VIEWS: list[tuple[str, float, float, float]] = [
    ('original', 0.0, 1.0, 1.0),
    ('rotate +10', 10.0, 1.0, 1.0),
    ('rotate -10', -10.0, 1.0, 1.0),
    ('zoom in 1.10', 0.0, 1.10, 1.0),
    ('zoom out 0.92', 0.0, 0.92, 1.0),
    ('darker 0.8', 0.0, 1.0, 0.8),
    ('brighter 1.2', 0.0, 1.0, 1.2),
]
NO_TTA = TTA_VIEWS[:1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Label game screenshots with trained Task 4 models.')
    parser.add_argument('--models', type=Path,
                        help='a folder of *.pt checkpoints, or one .pt file')
    parser.add_argument('--images', type=Path, help='folder of images to label (searched recursively)')
    parser.add_argument('--out', type=Path, default=Path('task4_predictions_new.csv'), help='output csv')
    parser.add_argument('--batch-size', type=int, default=64, help='images per GPU batch (lower it if out of memory)')
    parser.add_argument('--tta', action='store_true',
                        help=f'average {len(TTA_VIEWS)} fixed views per image (see TTA_VIEWS); {len(TTA_VIEWS)}x slower')
    parser.add_argument('--pack', type=Path,
                        help='instead of predicting: copy the best.pt of each run in --members-file into this folder')
    parser.add_argument('--members-file', type=Path, help='text file with one run name per line (for --pack)')
    return parser.parse_args()


def pack_models(target: Path, members_file: Path) -> None:
    """Copy best.pt of each listed run to <target>/<run>.pt, the folder a teammate needs."""
    runs = [line.strip() for line in members_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    target.mkdir(parents=True, exist_ok=True)
    for run in runs:
        source = EXP_ROOT / run / 'best.pt'
        if not source.is_file():
            raise SystemExit(f'{source} not found')
        shutil.copy2(source, target / f'{run}.pt')
    total_mb = sum((target / f'{run}.pt').stat().st_size for run in runs) / 2 ** 20
    print(f'packed {len(runs)} models into {target} ({total_mb:,.0f} MB)')


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the architecture without downloading ImageNet weights, then load the trained weights."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)  # trusted team files only
    settings = checkpoint['args']
    model = models.get_model(settings['model'], weights=None)
    head_name = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)][-1]
    parent_name, _, attribute = head_name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, nn.Linear(getattr(parent, attribute).in_features, NUM_CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device, memory_format=torch.channels_last).eval()
    return model, settings


def list_images(folder: Path) -> list[Path]:
    images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f'no images ({", ".join(sorted(IMAGE_SUFFIXES))}) found under {folder}')
    return images


def apply_view(batch: torch.Tensor, turn_degrees: float, zoom: float, brightness: float) -> torch.Tensor:
    """One fixed TTA view of a [0, 1] batch; rotation and zoom use the same maths as
    train.strong_augment (aspect-corrected rotation, black outside the picture)."""
    if brightness != 1.0:
        batch = (batch * brightness).clamp(0, 1)
    if turn_degrees == 0.0 and zoom == 1.0:
        return batch
    size, _, height, width = batch.shape
    turn = torch.tensor(np.deg2rad(turn_degrees), device=batch.device)
    theta = torch.zeros(size, 2, 3, device=batch.device)
    theta[:, 0, 0] = torch.cos(turn) / zoom
    theta[:, 0, 1] = -torch.sin(turn) * height / width / zoom
    theta[:, 1, 0] = torch.sin(turn) * width / height / zoom
    theta[:, 1, 1] = torch.cos(turn) / zoom
    grid = F.affine_grid(theta, list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def probabilities_of_batches(model: nn.Module, settings: dict, batches, device: torch.device,
                             views: list[tuple[str, float, float, float]]) -> np.ndarray:
    """Softmax averaged over `views` for each uint8 (N, 3, H, W) batch from `batches`."""
    preprocess = Preprocess(device, settings['height'], settings['width'], 0.0, settings.get('aug', 'basic'))
    use_amp = device.type == 'cuda'
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_u8 in batches:
            # Same resize as train.Preprocess (bilinear, align_corners=False), then each view, then normalize.
            resized = F.interpolate(batch_u8.to(device).float().div_(255), size=preprocess.size,
                                    mode='bilinear', align_corners=False)
            total = None
            for _, turn, zoom, brightness in views:
                inputs = ((apply_view(resized, turn, zoom, brightness) - preprocess.mean) / preprocess.std
                          ).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    probabilities = F.softmax(model(inputs).float(), dim=1)
                total = probabilities if total is None else total + probabilities
            chunks.append((total / len(views)).cpu().numpy())
    return np.concatenate(chunks)


def image_batches(image_paths: list[Path], batch_size: int):
    """uint8 batches from image files; images are stacked only when they share a size."""
    for begin in range(0, len(image_paths), batch_size):
        arrays = [np.asarray(Image.open(path).convert('RGB')) for path in image_paths[begin:begin + batch_size]]
        if len({array.shape for array in arrays}) == 1:
            yield torch.from_numpy(np.ascontiguousarray(np.stack(arrays).transpose(0, 3, 1, 2)))
        else:
            for array in arrays:  # mixed sizes: one image at a time
                yield torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None]


def model_probabilities(model: nn.Module, settings: dict, image_paths: list[Path], device: torch.device,
                        batch_size: int, views: list[tuple[str, float, float, float]] = NO_TTA) -> np.ndarray:
    return probabilities_of_batches(model, settings, image_batches(image_paths, batch_size), device, views)


def main() -> None:
    args = parse_args()
    if args.pack:
        if not args.members_file:
            raise SystemExit('--pack needs --members-file')
        pack_models(args.pack, args.members_file)
        return
    if not args.models or not args.images:
        raise SystemExit('give --models and --images (or --pack with --members-file)')
    checkpoints = sorted(args.models.glob('*.pt')) if args.models.is_dir() else [args.models]
    if not checkpoints:
        raise SystemExit(f'no .pt files in {args.models}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_paths = list_images(args.images)
    views = TTA_VIEWS if args.tta else NO_TTA
    print(f'{len(image_paths):,} images, {len(checkpoints)} models, {len(views)} view(s) per image, '
          f'device {device}', flush=True)
    probability_sum = np.zeros((len(image_paths), NUM_CLASSES))
    for number, checkpoint_path in enumerate(checkpoints, 1):
        start = time.perf_counter()
        model, settings = load_model(checkpoint_path, device)
        probability_sum += model_probabilities(model, settings, image_paths, device, args.batch_size, views)
        print(f'  [{number}/{len(checkpoints)}] {checkpoint_path.stem} ({settings["model"]}, '
              f'{settings["width"]}x{settings["height"]}) {time.perf_counter() - start:.0f}s', flush=True)
        del model
    probabilities = probability_sum / len(checkpoints)
    predicted = probabilities.argmax(1)
    table = pd.DataFrame({'file_name': [str(p.relative_to(args.images)) for p in image_paths],
                          'prediction': predicted, 'game': [TEAM_LABELS[i] for i in predicted]})
    for index, name in enumerate(CLASS_NAMES):
        table[f'prob_{name}'] = probabilities[:, index].round(5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
