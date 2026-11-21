"""Task 4: how well finished runs hold up when their validation images are damaged.

The scene split still scores far above the leaderboard, so this is a second, train-only check
of robustness. Each run's own validation images (never trained on) get one fixed kind of damage
at a time and the macro F1 is measured again. The damage types are generic photo/screen faults
chosen before looking at any result and not modelled on the test images (TA ruling 15 Nov).
Some types also appear in --aug robust training; the table marks them so the reader can tell
a fair comparison (types no recipe trained on) from a partly circular one.

    python Task4/stress.py --runs cnt_sc_f0,cntRS_sc_f0,cnsRS2_sc_f0,ev2sRS_sc_f0
    python Task4/stress.py --runs ... --mix ens=cntRS_sc_f0+cnsRS2_sc_f0+ev2sRS_sc_f0

Logits are cached per run (stress_logits.npy, stress.json) so ensembles can be scored later
without the GPU. Uses final.pt (last epoch) when it exists, else best.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as t4  # noqa: E402

# name -> which recipes saw this kind of damage in training ('strong' = colour/blur/noise/turn,
# 'robust' adds low resolution, JPEG, grey, stretch/shift/crop and cutout). '-' = none of them.
DAMAGE_SEEN = {
    'clean': '-',
    'jpeg15': 'robust',
    'lowres0.3': 'robust',
    'blur2': 'strong',
    'noise0.1': 'strong',
    'dark0.35': 'strong',
    'contrast0.35': 'strong',
    'turn25': 'strong',
    'crop1.5': 'robust',
    'stretch1.4': 'robust',
    'motion11': '-',
    'posterize3': '-',
    'haze0.5': '-',
    'vignette': '-',
    'sharpen': '-',
    'cover0.4': 'robust',
}
DAMAGES = list(DAMAGE_SEEN)


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return weights / weights.sum()


def separable_blur(batch: torch.Tensor, row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
    channels = batch.shape[1]
    row = row.to(batch).view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    column = column.to(batch).view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    batch = F.conv2d(F.pad(batch, (row.shape[-1] // 2,) * 2 + (0, 0), mode='replicate'), row, groups=channels)
    return F.conv2d(F.pad(batch, (0, 0) + (column.shape[-2] // 2,) * 2, mode='replicate'), column, groups=channels)


def stretch(batch: torch.Tensor, zoom_x: float, zoom_y: float) -> torch.Tensor:
    theta = torch.tensor([[1 / zoom_x, 0.0, 0.0], [0.0, 1 / zoom_y, 0.0]], device=batch.device, dtype=batch.dtype)
    grid = F.affine_grid(theta.expand(batch.shape[0], 2, 3), list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def damage(batch: torch.Tensor, kind: str, generator: torch.Generator) -> torch.Tensor:
    """One fixed kind of damage on a float batch in [0, 1] at model input size. Deterministic per batch."""
    height, width = batch.shape[2], batch.shape[3]
    gray = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
    if kind == 'clean':
        return batch
    if kind == 'jpeg15':
        from torchvision.io import decode_jpeg, encode_jpeg
        as_bytes = (batch * 255).round().clamp(0, 255).to(torch.uint8)
        encoded = [e.cpu() for e in encode_jpeg(list(as_bytes), quality=15)]
        return torch.stack(decode_jpeg(encoded, device=batch.device)).float().div(255)
    if kind == 'lowres0.3':
        small = F.interpolate(batch, size=(int(height * 0.3), int(width * 0.3)), mode='area')
        return F.interpolate(small, size=(height, width), mode='bilinear', align_corners=False)
    if kind == 'blur2':
        kernel = gaussian_kernel(2.0)
        return separable_blur(batch, kernel, kernel)
    if kind == 'noise0.1':
        noise = torch.randn(batch.shape, generator=generator, device='cpu').to(batch.device)
        return (batch + 0.1 * noise).clamp(0, 1)
    if kind == 'dark0.35':
        return batch * 0.35
    if kind == 'contrast0.35':
        mean = gray.mean(dim=(1, 2, 3), keepdim=True)
        return ((batch - mean) * 0.35 + mean).clamp(0, 1)
    if kind == 'turn25':
        return t4.affine(batch, 25 * np.pi / 180, 1.0)
    if kind == 'crop1.5':
        return t4.affine(batch, 0.0, 1.5)
    if kind == 'stretch1.4':
        return stretch(batch, 1.4, 1.0)
    if kind == 'motion11':
        return separable_blur(batch, torch.full((11,), 1 / 11), torch.ones(1))
    if kind == 'posterize3':
        return torch.floor(batch * 8).clamp(max=7) / 7
    if kind == 'haze0.5':
        low = torch.rand((batch.shape[0], 1, 4, 8), generator=generator, device='cpu').to(batch.device)
        fog = F.interpolate(low, size=(height, width), mode='bicubic', align_corners=False).clamp(0, 1)
        fog = 0.6 + 0.4 * fog
        return batch * 0.5 + fog * 0.5
    if kind == 'vignette':
        ys = torch.linspace(-1, 1, height, device=batch.device).view(1, 1, -1, 1)
        xs = torch.linspace(-1, 1, width, device=batch.device).view(1, 1, 1, -1)
        return batch * (1 - 0.8 * ((xs ** 2 + ys ** 2) / 2).clamp(0, 1))
    if kind == 'sharpen':
        kernel = gaussian_kernel(1.5)
        return (batch + 1.5 * (batch - separable_blur(batch, kernel, kernel))).clamp(0, 1)
    if kind == 'cover0.4':
        out = batch.clone()
        ph, pw = int(height * 0.4), int(width * 0.4)
        tops = torch.randint(0, height - ph + 1, (batch.shape[0],), generator=generator)
        lefts = torch.randint(0, width - pw + 1, (batch.shape[0],), generator=generator)
        for index, (top, left) in enumerate(zip(tops.tolist(), lefts.tolist())):
            out[index, :, top:top + ph, left:left + pw] = 0.5
        return out
    raise ValueError(f'unknown damage {kind}')


def run_logits(run_name: str, device: torch.device, images: torch.Tensor, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """(damages, images, 5) logits on the run's validation images, cached in the run folder."""
    run_dir = t4.EXP_ROOT / run_name
    cache, info_file = run_dir / 'stress_logits.npy', run_dir / 'stress.json'
    val_idx = np.load(run_dir / 'val_idx.npy')
    if limit:
        val_idx = val_idx[:limit]
    if cache.is_file() and info_file.is_file() and not limit:
        info = json.loads(info_file.read_text(encoding='utf-8'))
        if info['damages'] == DAMAGES:
            return np.load(cache), val_idx
    which = 'final' if (run_dir / 'final.pt').is_file() else 'best'
    model, settings, epoch = t4.load_run_model(run_dir, device, which)
    preprocess = t4.Preprocess(device, settings['height'], settings['width'], 0.0)
    piece = (settings.get('micro_batch') or settings['batch_size']) * 2
    mean, std = preprocess.mean, preprocess.std
    print(f'=== {run_name} ({settings["model"]}, {which}.pt epoch {epoch}, {len(val_idx):,} images) ===', flush=True)
    all_logits = []
    for kind in DAMAGES:
        start = time.perf_counter()
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(val_idx), piece):
                generator = torch.Generator().manual_seed(first)  # same damage for every run
                chunk = torch.from_numpy(val_idx[first:first + piece])
                batch = images[chunk].to(device).float().div_(255)
                batch = F.interpolate(batch, size=preprocess.size, mode='bilinear', align_corners=False)
                batch = ((damage(batch, kind, generator) - mean) / std).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    outputs.append(model(batch).float().cpu())
        all_logits.append(torch.cat(outputs).numpy())
        print(f'      {kind:13s} ({time.perf_counter() - start:.0f}s)', flush=True)
    stacked = np.stack(all_logits).astype(np.float32)
    if not limit:
        np.save(cache, stacked)
        info_file.write_text(json.dumps({'damages': DAMAGES, 'checkpoint': which, 'epoch': epoch}), encoding='utf-8')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return stacked, val_idx


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Task 4 robustness to damaged validation images')
    parser.add_argument('--runs', required=True, help='comma list of finished runs')
    parser.add_argument('--mix', action='append', default=[],
                        help='label=runA+runB (runs that share validation images); prints an averaged row')
    parser.add_argument('--limit', type=int, default=0, help='only the first N validation images (smoke test)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--data-root', default='')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.data_root:
        t4.DATA_DIR = Path(args.data_root).resolve()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    train_frame = pd.read_csv(t4.DATA_DIR / 'train.csv')
    labels = train_frame['label'].to_numpy(dtype=int)
    images, _ = t4.load_train_images(train_frame)
    runs = [name for name in args.runs.split(',') if name]
    probs, val_sets = {}, {}
    for name in runs:
        logits, val_idx = run_logits(name, device, images, args.limit)
        probs[name], val_sets[name] = softmax(logits), val_idx
    rows = [(name, probs[name], val_sets[name]) for name in runs]
    for spec in args.mix:
        label, _, members = spec.partition('=')
        members = members.split('+')
        for member in members[1:]:
            if not np.array_equal(val_sets[member], val_sets[members[0]]):
                raise SystemExit(f'{member} holds out other images than {members[0]}')
        rows.append((label, np.mean([probs[m] for m in members], axis=0), val_sets[members[0]]))

    print('\nmacro F1 on damaged validation images (seen in training by: ' +
          ', '.join(f'{k}={v}' for k, v in DAMAGE_SEEN.items() if k != 'clean') + ')')
    header = f'{"run":24s}' + ''.join(f'{d[:10]:>11s}' for d in DAMAGES) + f'{"mean dmg":>10s}{"unseen":>9s}'
    print(header)
    unseen = [i for i, d in enumerate(DAMAGES) if DAMAGE_SEEN[d] == '-' and d != 'clean']
    for label, run_probs, val_idx in rows:
        scores = [t4.macro_f1(labels[val_idx], run_probs[i].argmax(1)) for i in range(len(DAMAGES))]
        print(f'{label:24s}' + ''.join(f'{s:11.4f}' for s in scores) +
              f'{np.mean(scores[1:]):10.4f}{np.mean([scores[i] for i in unseen]):9.4f}')


if __name__ == '__main__':
    main()
