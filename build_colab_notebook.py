"""Generate the Colab notebook from the current repository workflow."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def markdown(text):
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell['id'] = f'colab-text-{len(cells):02d}'
    cells.append(cell)


def code(tag, text):
    cell = nbf.v4.new_code_cell(text.strip())
    cell.metadata['tags'] = [tag]
    cell['id'] = 'colab-' + tag
    cells.append(cell)


markdown('''# Player Intelligence System on Google Colab

This notebook runs the repository implementation for the five competition tasks. Upload the competition data to Google Drive as a ZIP file or choose an existing dataset folder. Task 4 requires the separately distributed image dataset and benefits from a Colab GPU. Output files are saved to a run directory on Drive by default.

The notebook clones the repository's main branch once and reports the commit used. Existing checkouts are reused without updating or discarding changes. Local validation scores are not leaderboard scores. Confirm the submission row alignment, Task 4 labels, and Task 5 test-data policy with the teaching team.''')

code('clone', '''from pathlib import Path
import os
import subprocess
import sys

REPO_URL = 'https://github.com/Kawi-HBLI/player-intelligence-system-cpe342.git'
REPO_DIR = Path('/content/player-intelligence-system-cpe342')
if REPO_DIR.exists():
    if not (REPO_DIR / '.git').exists():
        raise RuntimeError(f'{REPO_DIR} already exists but is not a Git checkout. Choose a different path or move it manually.')
    origin = subprocess.check_output(['git', '-C', str(REPO_DIR), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin.rstrip('/').removesuffix('.git') != REPO_URL.removesuffix('.git'):
        raise RuntimeError(f'Existing checkout has an unexpected origin: {origin}')
    print('Reusing existing checkout without changing it.')
else:
    subprocess.run(['git', 'clone', '--branch', 'main', REPO_URL, str(REPO_DIR)], check=True)
print('Repository HEAD:', subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip())
status = subprocess.check_output(['git', '-C', str(REPO_DIR), 'status', '--short'], text=True).strip()
print('Working tree:', status if status else 'clean')
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))''')

code('install', '''subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(REPO_DIR / 'requirements-colab.txt')], check=True)''')

markdown('''## Settings

For `drive`, provide a ZIP or dataset folder in Drive. For `upload`, choose one ZIP in the upload dialog. For `localfolder`, provide an existing folder in the Colab runtime. ZIP archives are extracted to `/content`, so image training does not repeatedly read compressed files from Drive.

Set `RUN_NAME` to an existing run directory to inspect it after reconnecting. Training runs every selected task when enabled and does not automatically resume or skip existing outputs. Use a new run name to keep experiments separate. Peer prediction CSV files may be copied into the same run directory before assembly.''')

code('settings', '''USE_DRIVE = True
DATA_SOURCE = 'drive'  # 'drive', 'upload', or 'localfolder'
DATA_PATH = '/content/drive/MyDrive/CPE342/data.zip'
OUTPUT_ROOT = '/content/drive/MyDrive/CPE342/runs'
RUN_NAME = ''  # Empty creates a unique run; a name selects that run directory.
TASKS = [1, 2, 3, 4, 5]
PROFILE = 'balanced'
FOLDS = 3
SEED = 342
THREADS = min(4, os.cpu_count() or 2)
RUN_TRAINING = True
BATCH_SIZE = 128
WORKERS = 2
CONTAMINATION = 0.15
DOWNLOAD_RESULTS = False''')

code('data', '''from pis.notebook_setup import prepare_dataset, create_run_directory, archive_results

if DATA_SOURCE not in {'drive', 'upload', 'localfolder'}:
    raise ValueError('DATA_SOURCE must be drive, upload, or localfolder')
if any(task not in (1, 2, 3, 4, 5) for task in TASKS) or len(set(TASKS)) != len(TASKS):
    raise ValueError('TASKS must contain unique task numbers from 1 to 5')
if RUN_TRAINING and not TASKS:
    raise ValueError('Select at least one task when training is enabled')
if PROFILE not in {'quick', 'balanced', 'strong'} or FOLDS < 2:
    raise ValueError('Choose a valid profile and at least two folds')
if not USE_DRIVE and Path(OUTPUT_ROOT).expanduser().resolve().is_relative_to(Path('/content/drive')):
    raise ValueError('OUTPUT_ROOT is on Drive. Set USE_DRIVE=True or choose a local output directory such as /content/pis_runs.')
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
if DATA_SOURCE == 'drive' and not USE_DRIVE:
    raise ValueError('Set USE_DRIVE=True to use Drive as the data source')
if DATA_SOURCE == 'upload':
    from google.colab import files
    upload_dir = Path('/content/pis_uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(upload_dir)
        uploads = files.upload()
    finally:
        os.chdir(previous_cwd)
    if len(uploads) != 1:
        raise ValueError('Select exactly one ZIP archive')
    source = upload_dir / next(iter(uploads))
    if source.suffix.lower() != '.zip':
        raise ValueError('The uploaded file must be a ZIP archive')
else:
    source = Path(DATA_PATH).expanduser()
    if DATA_SOURCE == 'localfolder' and not source.is_dir():
        raise ValueError('For localfolder, DATA_PATH must be an existing directory')
DATA_ROOT = prepare_dataset(source, Path('/content/pis_data'))
OUTPUT_DIR = create_run_directory(Path(OUTPUT_ROOT).expanduser(), RUN_NAME or None)
print('Data root:', DATA_ROOT)
print('Run directory:', OUTPUT_DIR)''')

code('image_setup', '''from pis.common import image_folder

IMAGE_ROOT = None
IMAGE_DATA_ROOT = Path('/content/pis_image_data')
training_images = RUN_TRAINING and 4 in TASKS
assembly_ready = all((OUTPUT_DIR / f'task{task}_predictions.csv').is_file() for task in range(1, 6))
try:
    candidate = image_folder(DATA_ROOT)
    metadata_ready = all((candidate / f'{split}.csv').is_file() for split in ('train', 'test'))
    image_dirs = [(candidate / 'data' / split if (candidate / 'data' / split).is_dir()
                   else candidate / split) for split in ('train', 'test')]
    images_ready = all(folder.is_dir() and any(p.is_file() for p in folder.iterdir()) for folder in image_dirs)
except FileNotFoundError:
    metadata_ready = images_ready = False
if (training_images and not (metadata_ready and images_ready)) or (assembly_ready and not metadata_ready):
    print('Preparing the separately distributed Task 4 image dataset.')
    subprocess.run([sys.executable, str(REPO_DIR / 'prepare_images.py'), '--download',
                    '--data-root', str(IMAGE_DATA_ROOT)], check=True)
    candidate = image_folder(IMAGE_DATA_ROOT)
    metadata_ready = True
if metadata_ready:
    IMAGE_ROOT = candidate
if training_images:
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise RuntimeError('Task 4 requires working Colab torch and torchvision installations.') from exc
    print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__,
          'CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        print('Task 4 will use the CPU. Select a GPU runtime in Colab to accelerate training.')''')

code('audit', '''import json
from pis.audit import audit_data
from pis.common import environment, write_json

if RUN_TRAINING or not (OUTPUT_DIR / 'run_environment.json').exists():
    write_json(OUTPUT_DIR / 'run_environment.json', {
    'environment': environment(),
    'repository_head': subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip(),
    'working_tree_status': status,
    'settings': {'tasks': TASKS, 'profile': PROFILE, 'folds': FOLDS, 'seed': SEED,
                 'threads': THREADS, 'batch_size': BATCH_SIZE, 'workers': WORKERS,
                 'contamination': CONTAMINATION, 'run_training': RUN_TRAINING},
})
audit = audit_data(DATA_ROOT, OUTPUT_DIR, image_root=IMAGE_ROOT)
print('Data audit completed. Warnings:', audit.get('warnings', []))''')

code('train', '''from pis.tabular import train_task
from pis.image import train_image
from pis.anomaly import train_anomaly

if RUN_TRAINING:
    for task in TASKS:
        print(f'Training Task {task}')
        if task in (1, 2, 3):
            train_task(task, DATA_ROOT, OUTPUT_DIR, folds=FOLDS, seed=SEED,
                       profile=PROFILE, threads=THREADS)
        elif task == 4:
            train_image(IMAGE_ROOT, OUTPUT_DIR, seed=SEED, profile=PROFILE,
                        batch_size=BATCH_SIZE, workers=WORKERS)
        else:
            train_anomaly(DATA_ROOT, OUTPUT_DIR, seed=SEED, profile=PROFILE,
                          contamination=CONTAMINATION, threads=THREADS)
else:
    print('Training is disabled. Existing results remain available for inspection.')''')

markdown('''## Results and submission

Task 1 threshold tuning uses the same out-of-fold labels as the tuned score, so that score is optimistic. Task 4 validation is a single holdout. Task 5 has no labels, so local F3 cannot be calculated; its anomaly fraction is an assumption. Verify the competition rules before submitting.''')

code('results', '''for task in range(1, 6):
    metrics_path = OUTPUT_DIR / f'task{task}_metrics.json'
    if metrics_path.exists():
        print(f'\\nTask {task} metrics')
        print(json.dumps(json.loads(metrics_path.read_text(encoding='utf-8')), indent=2))
    else:
        print(f'Task {task}: no metrics file in this run.')
missing = [f'task{task}_predictions.csv' for task in range(1, 6)
           if not (OUTPUT_DIR / f'task{task}_predictions.csv').is_file()]
if missing:
    print('Submission not assembled. Missing:', ', '.join(missing))
else:
    if IMAGE_ROOT is None:
        print('Preparing Task 4 metadata required to validate the submission.')
        subprocess.run([sys.executable, str(REPO_DIR / 'prepare_images.py'), '--download',
                        '--data-root', str(IMAGE_DATA_ROOT)], check=True)
        IMAGE_ROOT = image_folder(IMAGE_DATA_ROOT)
    from pis.submission import assemble
    manifest = assemble(DATA_ROOT, OUTPUT_DIR, image_root=IMAGE_ROOT)
    print('Submission assembled and locally validated:', OUTPUT_DIR / 'submission.csv')
    print('Competition evaluator acceptance and ID alignment remain unverified.')''')

code('archive', '''try:
    archive = archive_results(OUTPUT_DIR)
except FileNotFoundError as exc:
    print('No completed results are available to archive:', exc)
else:
    print('Shareable results archive:', archive)
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(archive))''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'colab': {'name': 'Player_Intelligence_Colab.ipynb'},
})
nbf.validate(nb)
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source, '<colab-cell>', 'exec')
nbf.write(nb, ROOT / 'Player_Intelligence_Colab.ipynb')
print('Generated Player_Intelligence_Colab.ipynb')
