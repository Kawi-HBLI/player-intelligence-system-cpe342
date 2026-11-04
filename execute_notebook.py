"""Execute the read-only notebook view with the selected Python environment."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent


def main():
    local=ROOT/'.jupyter'
    spec=local/'kernels'/'pis-local'
    spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                    'display_name':'Player Intelligence local','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(local)
    runtime=local/'runtime';runtime.mkdir(exist_ok=True)
    temp=local/'tmp';temp.mkdir(exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp)
    path=ROOT/'Player_Intelligence_System.ipynb'
    notebook=nbformat.read(path,as_version=4)
    # This verification must not start another model training run.
    if not any('RUN_TRAINING = False' in c.source for c in notebook.cells if c.cell_type=='code'):
        raise ValueError('Notebook execution verification requires RUN_TRAINING=False')
    client=NotebookClient(notebook,timeout=300,kernel_name='pis-local',resources={'metadata':{'path':str(ROOT)}})
    client.execute(cwd=str(ROOT))
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    print('Executed all notebook cells successfully:',path)


if __name__=='__main__':main()
