"""Bundle reviewed source and measured summaries, excluding data and environments."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.ipynb'))
    files+=list((ROOT/'pis').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    files += [ROOT/'requirements.txt',ROOT/'.gitignore']
    out=ROOT/'outputs'/'main'
    files+=list(out.glob('*metrics.json'))
    files += [out/name for name in ('RESULTS_TH.md','validation_results.png','results_summary.json','data_audit.json','environment_verified.json')]
    missing=[p.name for p in files if not p.is_file()]
    if missing:raise FileNotFoundError(f'Cannot package incomplete outputs: {missing}')
    path=ROOT/'player-intelligence-source.zip'
    hashes={}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for file in sorted(set(files)):
            name=file.relative_to(ROOT).as_posix()
            hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
            bundle.write(file,name)
        bundle.writestr('BUNDLE_MANIFEST.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(path) as bundle:
        assert bundle.testzip() is None
        assert not any('.venv/' in name or '.cache/' in name for name in bundle.namelist())
    print(path)
    print('Files:',len(files),'ZIP bytes:',path.stat().st_size)


if __name__=='__main__':main()
