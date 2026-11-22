"""Task 4: where do out-of-fold mistakes happen? (training images only)

For every training image it measures simple picture properties (brightness, contrast,
saturation, share of black border from rotation, grain noise) and reports the OOF error
rate per fifth of each property, plus the most confident mistakes as a picture sheet.

    python Task4/error_analysis.py --group cnt=cnt_nc_f{fold} --views flip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oof  # noqa: E402
import train as t4  # noqa: E402


def picture_properties(images: torch.Tensor) -> pd.DataFrame:
    rows = []
    for start in range(0, len(images), 2048):
        x = images[start:start + 2048].float().div(255)
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        black = (x.max(1).values < 0.03).float().mean((1, 2))
        saturation = (x.max(1).values - x.min(1).values).mean((1, 2))
        # Grain: difference between the image and a 3x3 blurred copy.
        blurred = F.avg_pool2d(gray[:, None], 3, stride=1, padding=1)[:, 0]
        noise = (gray - blurred).abs().mean((1, 2))
        rows.append(pd.DataFrame({'brightness': gray.mean((1, 2)).numpy(), 'contrast': gray.std((1, 2)).numpy(),
                                  'saturation': saturation.numpy(), 'black_share': black.numpy(),
                                  'noise': noise.numpy()}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Task 4 OOF error analysis')
    parser.add_argument('--group', action='append', default=[])
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--views', default='none')
    parser.add_argument('--out', default=str(t4.EXP_ROOT / 'error_analysis'))
    parser.add_argument('--data-root', default='')
    args = parser.parse_args()
    if args.data_root:
        t4.DATA_DIR = Path(args.data_root).resolve()
    train_frame = pd.read_csv(t4.DATA_DIR / 'train.csv')
    labels = train_frame['label'].to_numpy(dtype=int)
    images, _ = t4.load_train_images(train_frame)
    folds = [int(f) for f in args.folds.split(',')]
    views = oof.PRESET_VIEW_SETS.get(args.views, args.views.split(','))
    groups = [oof.Group(*spec.split('=', 1), folds, len(labels)) for spec in args.group]
    probs = np.mean([g.oof(views) for g in groups], axis=0)
    covered = ~np.isnan(probs[:, 0])
    predicted = np.where(covered, np.nan_to_num(probs).argmax(1), -1)
    wrong = covered & (predicted != labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    props = picture_properties(images)
    props['label'], props['wrong'], props['covered'] = labels, wrong, covered
    props = props[props['covered']]
    print(f'OOF images {len(props):,}, wrong {int(props["wrong"].sum())} '
          f'({props["wrong"].mean():.2%})')
    print('\nerror rate per fifth of each picture property (training images only)')
    for column in ['brightness', 'contrast', 'saturation', 'black_share', 'noise']:
        bins = pd.qcut(props[column].rank(method='first'), 5, labels=['q1 low', 'q2', 'q3', 'q4', 'q5 high'])
        table = props.groupby(bins, observed=True).agg(images=('wrong', 'size'), error=('wrong', 'mean'),
                                                       low=(column, 'min'), high=(column, 'max'))
        cells = '  '.join(f'{name} {row.error:.2%} [{row.low:.3f}-{row.high:.3f}]' for name, row in table.iterrows())
        print(f'  {column:11s} {cells}')
    print('\nerror rate per game:')
    for c, name in enumerate(t4.TEAM_LABELS):
        part = props[props['label'] == c]
        print(f'  {name:10s} {part["wrong"].mean():.2%} of {len(part):,}')

    # Most confident mistakes: picture sheet with true / predicted game.
    worst = np.flatnonzero(wrong)
    worst = worst[np.argsort(-probs[worst].max(1))][:60]
    height, width = images.shape[2], images.shape[3]
    columns = 6
    sheet = Image.new('RGB', (columns * width, -(-len(worst) // columns) * (height + 14)), 'white')
    draw = ImageDraw.Draw(sheet)
    for k, index in enumerate(worst):
        left, top = (k % columns) * width, (k // columns) * (height + 14)
        sheet.paste(Image.fromarray(images[index].permute(1, 2, 0).numpy()), (left, top))
        draw.text((left + 2, top + height), f'{t4.TEAM_LABELS[labels[index]]}->{t4.TEAM_LABELS[predicted[index]]} '
                  f'{probs[index].max():.2f}', fill='black')
    sheet.save(out / 'worst_mistakes.jpg', quality=90)
    pd.DataFrame({'index': worst, 'file_name': train_frame['file_name'].to_numpy()[worst],
                  'true': labels[worst], 'predicted': predicted[worst],
                  'confidence': probs[worst].max(1)}).to_csv(out / 'worst_mistakes.csv', index=False)
    print(f'\nwrote {out / "worst_mistakes.jpg"} and worst_mistakes.csv')


if __name__ == '__main__':
    main()
