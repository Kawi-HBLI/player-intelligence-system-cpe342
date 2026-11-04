"""Train and evaluate the game screenshot classifier without using test labels."""
from pathlib import Path
import hashlib
import json
import random


class _Images:
    def __init__(self, paths, y, transform):
        self.paths, self.y, self.transform = paths, y, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        with Image.open(self.paths[idx]) as img:
            x = self.transform(img.convert('RGB'))
        return x, (-1 if self.y is None else int(self.y[idx]))


def train_image(data_root: Path, output_dir: Path, seed: int = 342,
                profile: str = 'balanced', batch_size: int = 128,
                epochs: int | None = None, workers: int = 0,
                pretrained: bool = True) -> dict:
    """Train ResNet18 on a duplicate-grouped split and predict the test set.

    CSV labels are preserved in the output; internal class indices are never
    interpreted as game names. No test image is used for training or selection.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms, models

    data_root, output_dir = Path(data_root), Path(output_dir)
    if profile not in {'quick', 'balanced', 'strong'}:
        raise ValueError('profile must be quick, balanced, or strong')
    epochs = epochs if epochs is not None else {'quick': 2, 'balanced': 5, 'strong': 10}[profile]
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError('epochs and batch_size must be positive and workers nonnegative')
    if not (data_root / 'train.csv').is_file():
        candidates = list(data_root.rglob('train.csv')) if data_root.exists() else []
        candidates = [p for p in candidates if (p.parent / 'test.csv').is_file()
                      and ((p.parent / 'data' / 'train').is_dir() or (p.parent / 'train').is_dir())]
        if len(candidates) != 1:
            raise FileNotFoundError(f'Expected a unique image dataset with train.csv, test.csv and data/train under {data_root}')
        data_root = candidates[0].parent
    for name in ('train.csv', 'test.csv'):
        if not (data_root / name).is_file():
            raise FileNotFoundError(f'Missing image metadata: {data_root / name}')
    train = pd.read_csv(data_root / 'train.csv', dtype={'id': str, 'player_id': str})
    test = pd.read_csv(data_root / 'test.csv', dtype={'id': str, 'player_id': str})
    def choose(columns, options, what):
        found = [c for c in options if c in columns]
        if not found:
            raise ValueError(f'Missing {what}; expected one of {options}, got {list(columns)}')
        return found[0]
    id_col = choose(train.columns, ('id', 'player_id', 'image_id', 'filename', 'file', 'image'), 'image identifier')
    if id_col not in test.columns:
        raise ValueError(f'Test metadata lacks identifier column {id_col}')
    if train[id_col].duplicated().any() or test[id_col].duplicated().any():
        raise ValueError('Training and test image identifiers must be unique within each file')
    label_col = choose(train.columns, ('class', 'label', 'target', 'prediction'), 'label')
    if train[id_col].isna().any() or test[id_col].isna().any() or train[label_col].isna().any():
        raise ValueError('Image identifiers and training labels must not be missing')
    classes = sorted(train[label_col].unique().tolist(), key=lambda x: (str(type(x)), str(x)))
    if len(classes) != 5:
        raise ValueError(f'Expected exactly 5 training classes, found {len(classes)}')
    class_to_index = {v: i for i, v in enumerate(classes)}
    labels = train[label_col].map(class_to_index).to_numpy(dtype=int)

    def image_paths(frame, split):
        folder = data_root / 'data' / split
        if not folder.is_dir():
            folder = data_root / split
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing image folder: {folder}')
        paths = []
        for value in frame[id_col]:
            name = str(value)
            path = folder / name
            if not path.suffix:
                path = folder / (name + '.jpg')
            if not path.is_file():
                raise FileNotFoundError(f'Missing {split} image for {value}: {path}')
            paths.append(path)
        return paths
    train_paths, test_paths = image_paths(train, 'train'), image_paths(test, 'test')
    # Duplicate bytes must remain on the same side of the validation split.
    hashes = []
    for path in train_paths:
        with path.open('rb') as handle:
            hashes.append(hashlib.sha256(handle.read()).hexdigest())
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=hashes))
    if len(set(labels[fit_idx])) != 5 or len(set(labels[val_idx])) != 5:
        raise ValueError('Cannot construct a grouped holdout containing all five classes')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([transforms.Resize((128, 256)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize((128, 256)), transforms.ToTensor(), normalize])
    def loader(paths, y, transform, shuffle=False):
        return DataLoader(_Images(paths, y, transform), batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=device.type == 'cuda',
                          generator=torch.Generator().manual_seed(seed))
    fit_loader = loader([train_paths[i] for i in fit_idx], labels[fit_idx], train_transform, True)
    val_loader = loader([train_paths[i] for i in val_idx], labels[val_idx], eval_transform)
    test_loader = loader(test_paths, None, eval_transform)
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': 3e-5},
        {'params': model.fc.parameters(), 'lr': 3e-4}], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'models' / 'task4_best.pt'
    best, history = -1.0, []
    def infer(data_loader):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for x, _ in data_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(x)
                predictions.extend(logits.argmax(1).cpu().tolist())
        return np.asarray(predictions, dtype=int)
    for epoch in range(epochs):
        # Keep frozen BatchNorm statistics unchanged during the head-only epoch.
        for name, param in model.named_parameters():
            param.requires_grad = epoch > 0 or name.startswith('fc.')
        model.train()
        if epoch == 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        loss_sum = 0.0
        for x, y in fit_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
        predicted = infer(val_loader)
        score = f1_score(labels[val_idx], predicted, average='macro', zero_division=0)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(fit_idx), 'validation_macro_f1': float(score)})
        print(f'Task 4 epoch {epoch + 1}/{epochs}: validation macro F1={score:.4f}', flush=True)
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'classes': classes, 'seed': seed,
                        'pretrained': pretrained, 'epoch': epoch + 1}, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    val_pred, test_pred = infer(val_loader), infer(test_loader)
    def original(indices):
        return [classes[int(i)] for i in indices]
    pd.DataFrame({'player_id': train.iloc[val_idx][id_col].to_numpy(),
                  'actual': train.iloc[val_idx][label_col].to_numpy(),
                  'prediction': original(val_pred)}).to_csv(output_dir / 'task4_validation.csv', index=False)
    pd.DataFrame({'player_id': test[id_col].to_numpy(), 'prediction': original(test_pred)}).to_csv(
        output_dir / 'task4_predictions.csv', index=False)
    metrics = {'task': 4, 'seed': seed, 'profile': profile, 'device': str(device),
               'pretrained': pretrained, 'train_size': int(len(fit_idx)), 'validation_size': int(len(val_idx)),
               'split': 'stratified 5-fold grouped by SHA256 image bytes; first fold held out',
               'classes': [v.item() if hasattr(v, 'item') else v for v in classes],
               'best_epoch': int(saved['epoch']), 'validation_macro_f1': float(best),
               'validation_accuracy': float(accuracy_score(labels[val_idx], val_pred)),
               'classification_report': classification_report(labels[val_idx], val_pred, labels=list(range(5)),
                                                                target_names=[str(v) for v in classes],
                                                                output_dict=True, zero_division=0),
               'confusion_matrix': confusion_matrix(labels[val_idx], val_pred, labels=list(range(5))).tolist(),
               'history': history}
    with (output_dir / 'task4_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics
