"""Run from a terminal or import the same functions from the teaching notebook."""
import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from pis.common import DEFAULT_DATA, DEFAULT_OUTPUT, environment, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPE342 Player Intelligence System')
    parser.add_argument('command', choices=('audit', 'train', 'assemble', 'validate'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--image-root', type=Path, help='Optional separate Task 4 dataset root')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tasks', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--profile', choices=('quick','balanced','strong'), default='balanced')
    parser.add_argument('--engine', choices=('auto','catboost','sklearn'), default='auto')
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=342)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--contamination', type=float, default=0.20)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--file', type=Path)
    args = parser.parse_args(argv)
    os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent / '.cache' / 'torch'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'audit':
        from pis.audit import audit_data
        result = audit_data(args.data_root, args.output_dir, args.image_root)
        print('Audit saved:', args.output_dir / 'data_audit.json')
        for warning in result['warnings']:
            print('NOTE:', warning)
    elif args.command == 'train':
        write_json(args.output_dir / ('run_tasks_' + '_'.join(map(str,args.tasks)) + '.json'), {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'environment': environment(), 'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
        for task in args.tasks:
            if task in (1,2,3):
                from pis.tabular import train_task
                result = train_task(task, args.data_root, args.output_dir, args.folds, args.seed,
                                    args.profile, args.threads, args.engine)
            elif task == 4:
                from pis.image import train_image
                result = train_image(args.image_root or args.data_root, args.output_dir, seed=args.seed, profile=args.profile,
                                     batch_size=args.batch_size, epochs=args.epochs, workers=args.workers)
            else:
                from pis.anomaly import train_anomaly
                result = train_anomaly(args.data_root, args.output_dir, args.seed, args.profile,
                                       args.contamination, args.threads)
            print(f'Task {task} completed: {result.get("oof_score", result.get("best_macro_f1", "see metrics JSON"))}')
    elif args.command == 'assemble':
        from pis.submission import assemble
        result = assemble(args.data_root, args.output_dir, args.image_root)
        print('Created:', result['file'])
    else:
        from pis.submission import validate_file
        result = validate_file(args.file or args.output_dir/'submission.csv', args.data_root, args.image_root)
        print('Local submission checks:', result)


if __name__ == '__main__':
    main()
