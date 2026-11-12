"""Download the instructor-provided image archive and safely unpack it."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile
import zlib
from pis.common import DEFAULT_DATA


def extract(archive, data_root):
    target = (Path(data_root) / '04_image_classification').resolve()
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for item in members:
            parts = PurePosixPath(item.filename).parts
            if parts and parts[0] == 'task4_image_classification':
                parts = parts[1:]
            # The 12 Nov archive has no top folder and carries macOS metadata plus its own
            # datacard.txt, which differs from the one Kaggle ships in the same folder.
            if not parts or parts[0] == '__MACOSX' or parts == ('datacard.txt',):
                continue
            dest = target.joinpath(*parts).resolve()
            if not dest.is_relative_to(target) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains unsafe member')
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            # Do not silently overwrite different competition metadata.
            if dest.exists():
                crc = 0
                with dest.open('rb') as existing:
                    for block in iter(lambda: existing.read(1024*1024), b''):
                        crc = zlib.crc32(block, crc)
                if dest.stat().st_size != item.file_size or (crc & 0xffffffff) != item.CRC:
                    raise FileExistsError(f'Existing file differs: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, dest.open('wb') as output:
                shutil.copyfileobj(source, output)
            count += 1
    print(f'Extracted {count} files into {target}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=DEFAULT_DATA.parent / 'task4_image_classification.zip')
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    if args.download and not args.archive.exists():
        import gdown
        gdown.download(id='1C8csylbBo3Oab24ODGKPVom3_SDUJ_Qw', output=str(args.archive), quiet=False)
    extract(args.archive, args.data_root)


if __name__ == '__main__':
    main()
