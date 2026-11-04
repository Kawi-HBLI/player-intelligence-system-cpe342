"""Rebuild the portable notebook from the verified source; no duplicate implementation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    cells = []
    def markdown(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                      'source':text.splitlines(keepends=True)})
    markdown('''# Player Intelligence System · CPE342

สมุดงานครบ 5 Task: ตรวจข้อมูล → ฝึกโมเดล → วิเคราะห์ข้อผิดพลาด → ตรวจไฟล์ส่ง

เริ่มจากอ่านผลที่มีอยู่ก่อน โดย `RUN_TRAINING=False` จะไม่ฝึกซ้ำ หากใช้บน Kaggle ครั้งแรกให้เปิด GPU และเปลี่ยนเป็น `True` เพื่อสร้างผลใหม่ ใช้ข้อมูลของการแข่งขันที่แนบไว้ใน Input และเปิด Internet เพื่อดาวน์โหลดน้ำหนัก ResNet18 ตามที่กติกา Task 4 อนุญาต หรือแนบน้ำหนักที่ดาวน์โหลดไว้แล้วและตั้ง TORCH_HOME

โค้ดทั้งหมดฝังอยู่ใน notebook นี้ จึงไม่ต้องแนบโฟลเดอร์ `pis` แยก เมื่อแก้ source ในเครื่องให้รัน `build_notebook.py` เพื่อสร้างฉบับที่ตรงกันอีกครั้ง

สิ่งที่ต้องรู้: Task 1 ขาด label 13,184 แถว ต้องแยกออก; Task 4 ใช้ชื่อเกมจาก CSV จริง; Task 5 ไม่มี label จึงไม่มีคะแนน F3 ในเครื่อง; การจับคู่ ANS กับแต่ละ Task ใช้ลำดับ test.csv ตามสมมติฐานที่ยังควรยืนยันกับ TA

อ้างอิง: [โจทย์ Kaggle](https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml/overview/description) · [สไลด์ผู้สอน](https://www.canva.com/design/DAHVJwyZgy0/Ec3RMMLdy0FgfOLfIb3PhA/view)
''')
    markdown('''## 1. ตั้งค่า

ในเครื่องให้เลือก Python จาก `solution/.venv` ถ้าขาดเครื่องมือให้ติดตั้ง `requirements.txt` ก่อน บน Kaggle มี PyTorch และ scikit-learn อยู่แล้ว หากไม่มี CatBoost โค้ดจะใช้ HistGradientBoosting แทน หรือรัน `%pip install catboost` ก่อนฝึก

`balanced` ใช้ CatBoost สูงสุด 1,000 รอบต่อ fold, ภาพ 5 epoch และ Isolation Forest 400 ต้นต่อมุมมอง `strong` เพิ่มงบฝึก แต่ไม่รับประกันว่าคะแนนดีขึ้น ต้องเปรียบเทียบบน split เดิม
''')
    code('''from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd

RUN_TRAINING = False
PROFILE = "balanced"
FOLDS = 3
SEED = 342
TASKS_TO_RUN = [1, 2, 3, 4, 5]
CONTAMINATION = 0.15  # สมมติฐานสัดส่วนผิดปกติ ยังไม่มี label ให้ปรับด้วย F3
DATA_ROOT = None  # กำหนด Path เองถ้าค้นเจอ dataset หลายชุด
IMAGE_ROOT = None  # ถ้าแนบภาพแยกจากตาราง ให้ชี้โฟลเดอร์ที่มี train.csv/test.csv ของภาพ

candidates = [Path.cwd(), Path.cwd()/"cpe342-kaggle-main", Path.cwd().parent/"cpe342-kaggle-main",
              Path.cwd()/"Player Intelligence System - CPE342 ML"/"cpe342-kaggle-main"]
if Path('/kaggle/input').exists():
    candidates += [p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')]
roots = sorted({p.resolve() for p in candidates if (p/'sample_submission.csv').exists()})
if DATA_ROOT is None:
    if len(roots) != 1:
        raise ValueError(f"ตั้ง DATA_ROOT ให้ชี้โฟลเดอร์ที่มี sample_submission.csv: พบ {roots}")
    DATA_ROOT = roots[0]
else:
    DATA_ROOT = Path(DATA_ROOT)
OUTPUT_DIR = (Path('/kaggle/working')/'outputs' if Path('/kaggle/working').exists()
              else DATA_ROOT.parent/'solution'/'outputs'/'main')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cache_base = Path('/kaggle/working') if Path('/kaggle/working').exists() else OUTPUT_DIR.parent.parent
os.environ.setdefault('TORCH_HOME', str(cache_base/'.cache'/'torch'))
if IMAGE_ROOT is None:
    IMAGE_ROOT = DATA_ROOT/'04_image_classification'
    if not ((IMAGE_ROOT/'train.csv').exists() and
            ((IMAGE_ROOT/'train').is_dir() or (IMAGE_ROOT/'data'/'train').is_dir())) and Path('/kaggle/input').exists():
        image_roots = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
                       if (p.parent/'test.csv').exists() and
                       ((p.parent/'train').is_dir() or (p.parent/'data'/'train').is_dir())]
        if len(image_roots)==1:
            IMAGE_ROOT=image_roots[0]
print('ข้อมูล:', DATA_ROOT, '\\nผลทดลอง:', OUTPUT_DIR)
print('ภาพ:', IMAGE_ROOT)
''')
    modules = {str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8')
               for p in sorted((ROOT/'pis').glob('*.py'))}
    code('''# Source snapshot: สร้าง package เดียวกับที่ใช้ทดสอบในเครื่อง
SOURCE_FILES = ''' + repr(modules) + '''
runtime = Path.cwd()/'notebook_runtime'
for name, content in SOURCE_FILES.items():
    path = runtime/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
sys.path.insert(0, str(runtime.resolve()))
from pis.audit import audit_data
from pis.tabular import train_task
from pis.anomaly import train_anomaly
from pis.image import train_image
from pis.submission import assemble, validate_file
''')
    markdown('''## 2. สำรวจข้อมูลก่อนเลือกโมเดล

ดูจำนวนตัวอย่าง ค่าที่หายไป สัดส่วนคลาส และข้อมูลเป้าหมาย ห้ามใช้ `player_id` เป็นตัวทำนาย และห้ามนำผู้เล่นคนละชุดมาเชื่อมเพราะแต่ละ Task มีรหัสคนละลำดับ

Task 3 มีผู้ไม่ใช้จ่ายจำนวนมากและหางยาว ให้ดูทั้งสัดส่วนศูนย์และการกระจาย ไม่ใช้ค่าเฉลี่ยเพียงอย่างเดียวอธิบายผู้เล่นทุกคน
''')
    code('''audit = audit_data(DATA_ROOT, OUTPUT_DIR, IMAGE_ROOT)
display(pd.DataFrame([{'task':task, 'split':split, 'rows':item['rows'],
                       'missing_target':item.get('missing_target_rows',0),
                       'missing_cells':sum(item['missing_per_column'].values())}
                      for task,info in audit['tasks'].items()
                      for split,item in info.items() if split in ('train','test')]))
for note in audit['warnings']:
    print('ข้อสังเกต:', note)
''')
    code('''import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,3,figsize=(15,4))
for task, ax in zip((1,2),axes[:2]):
    target = 'is_cheater' if task==1 else 'segment'
    folder = '01_binary_classification' if task==1 else '02_multiclass_classification'
    y = pd.read_csv(DATA_ROOT/folder/'train.csv',usecols=[target])[target]
    y.value_counts().sort_index().plot.bar(ax=ax,color='#236c9c')
    ax.set(title=f'Task {task}: labeled training class counts',xlabel='Class',ylabel='Players (count)')
y3 = pd.read_csv(DATA_ROOT/'03_regression'/'train.csv',usecols=['spending_30d']).spending_30d
axes[2].hist(np.log1p(y3),bins=50,color='#236c9c')
axes[2].set(title='Task 3: training spending distribution',xlabel='log(1 + spending in THB)',ylabel='Players (count)')
fig.suptitle('Source: provided training CSVs; log transform is for visualization only')
plt.tight_layout()
plt.show()
''')
    markdown('''## 3. ฝึกและตรวจสอบ Tasks 1–3

CatBoost รับทั้งค่าตัวเลขและหมวดหมู่ ช่วยลดขั้นตอนแปลงข้อมูล ใช้ cross-validation เพื่อให้ทุกแถวมีผลทำนายจากโมเดลที่ไม่ได้ฝึกบนแถวนั้น เรียกว่า OOF หากใช้โมเดลสำรอง การเติมค่าหายและแปลงหมวดหมู่จะเรียนรู้เฉพาะ fold ฝึก

Task 1 ใช้ macro F2: คำนวณ F2 แยกแต่ละคลาสแล้วเฉลี่ย ไม่เท่ากับ F2 เฉพาะคลาสโกง การเลือก threshold บน OOF แล้วรายงานคะแนนบน OOF เดิมมีความเอนเอียง คะแนน threshold 0.5 จึงแสดงไว้ด้วย

Task 2 ใช้ macro F1 เพื่อไม่ให้คลาสใหญ่กลบคลาสเล็ก

Task 3 ใช้คะแนน `1/(1 + sum(abs(y-yhat))/sum(y))` ยิ่งสูงยิ่งดี ถ้าทายศูนย์ทุกคนจะได้ 0.5 เมื่อยอดรวมจริงเป็นบวก ลด MAE บนหน่วยบาทจริงจึงตรงกับคะแนนนี้ การฝึก log-target แล้วแปลงกลับไม่รับประกันว่า MAE จะดีขึ้น
''')
    code('''if RUN_TRAINING:
    for task in (1,2,3):
        if task in TASKS_TO_RUN:
            result = train_task(task,DATA_ROOT,OUTPUT_DIR,folds=FOLDS,seed=SEED,
                                profile=PROFILE,threads=8)
            print('Task',task,'OOF:',result['oof_score'])
else:
    print('ใช้ผลเดิม; เปลี่ยน RUN_TRAINING=True เพื่อฝึกใหม่')
''')
    markdown('''## 4. Task 4: ภาพเกม

ใช้ ResNet18 ที่ฝึกล่วงหน้าตามสิทธิ์ของ Task 4 แล้วปรับกับภาพการแข่งขัน แยกภาพที่มี byte เหมือนกันให้อยู่ฝั่งเดียวกัน เพื่อลดคะแนนตรวจสอบที่สูงจากภาพซ้ำ เริ่มฝึกชั้นทำนายก่อน แล้วจึงปรับทั้งโมเดล เก็บ checkpoint ที่ macro F1 สูงสุด

ผลนี้เป็นคะแนนจาก holdout เดียวที่ใช้เลือก epoch จึงอาจสูงกว่าผลใหม่ได้ ภาพใกล้เคียงกันแต่ byte ต่างกันยังอาจข้าม split ควรตรวจต่อหากคะแนนสูงผิดปกติ ใช้ป้ายกำกับชื่อเกมตาม CSV จริง ไม่มีการเดาลำดับคลาสจากหน้า Overview
''')
    code('''if RUN_TRAINING and 4 in TASKS_TO_RUN:
    result = train_image(IMAGE_ROOT,OUTPUT_DIR,seed=SEED,profile=PROFILE,batch_size=128,workers=0)
    print('Image holdout macro F1:',result['validation_macro_f1'])
''')
    markdown('''## 5. Task 5: ความผิดปกติที่ไม่มีคำตอบกำกับ

ฝึก Isolation Forest สองมุมมอง ได้แก่ค่าพฤติกรรมดิบ และการเปลี่ยนแปลงข้าม 4 ช่วงเวลา รวมอันดับคะแนนเพื่อไม่ให้สเกลของโมเดลหนึ่งกลบอีกโมเดล การเลือก 15% ที่คะแนนสูงสุดเป็นสมมติฐานเริ่มต้น ไม่ใช่สัดส่วนความผิดปกติที่พิสูจน์แล้ว

งานนี้มีเฉพาะ test.csv ที่ไม่มี label จึงตีความตามโจทย์เฉพาะว่าใช้ข้อมูลนี้ฝึกแบบไม่มีผู้สอนได้ แต่สไลด์กติกาทั่วไปห้ามฝึกบน test ต้องยืนยันข้อยกเว้นกับ TA ก่อนส่งงานจริง คะแนน F3 จะทราบจากการประเมินที่มีคำตอบจริงเท่านั้น
''')
    code('''if RUN_TRAINING and 5 in TASKS_TO_RUN:
    result = train_anomaly(DATA_ROOT,OUTPUT_DIR,seed=SEED,profile=PROFILE,
                           contamination=CONTAMINATION,threads=8)
    print('Flagged:',result['flagged_rows'],'Local F3:',result['macro_f3'])
''')
    markdown('''## 6. อ่านผลและตรวจข้อผิดพลาด

อย่าเทียบคะแนนคนละ Task ราวกับเป็น metric เดียวกัน และยังคำนวณคะแนนรวมที่เทียบ Kaggle ไม่ได้ เพราะไม่มีคำตอบ Task 5 และยังไม่ทราบน้ำหนักคะแนนที่แน่นอน ตรวจ confusion matrix ต่อคลาสและตัวอย่างที่ผิด แล้วเลือกการทดลองถัดไปจากหลักฐาน
''')
    code('''metrics = {}
for task in range(1,6):
    path = OUTPUT_DIR/f'task{task}_metrics.json'
    if path.exists():
        metrics[task] = json.loads(path.read_text(encoding='utf-8'))
display(pd.DataFrame([{'task':t, 'metric':m.get('metric','macro_f1' if t==4 else 'macro_f3'),
                        'score':m.get('fixed_0_5_oof_score',m.get('oof_score',m.get('validation_macro_f1'))),
                        'evaluation':'OOF fixed threshold' if t==1 else 'OOF' if t<4 else 'holdout' if t==4 else 'unlabeled'}
                       for t,m in metrics.items()]))
for task,m in metrics.items():
    if 'confusion_matrix' not in m:
        continue
    matrix=np.array(m['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5,4))
    im=ax.imshow(matrix,cmap='Blues')
    for (r,c),value in np.ndenumerate(matrix):
        ax.text(c,r,str(value),ha='center',va='center')
    caption = f"OOF, tuned threshold {m['threshold']:.3f}" if task==1 else 'OOF' if task==2 else 'image holdout'
    names = m.get('classes',list(range(len(matrix))))
    ax.set(title=f'Task {task}: confusion matrix ({caption})',xlabel='Predicted class',ylabel='True class')
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right')
    ax.set_yticks(range(len(names)),names)
    fig.colorbar(im,ax=ax,label='Players/images (count)')
    plt.show()
''')
    markdown('''### โมเดลใช้ข้อมูลใด และผิดพลาดกับใคร

ความสำคัญของคุณลักษณะช่วยเสนอสมมติฐานสำหรับทดลองต่อ แต่ไม่พิสูจน์สาเหตุทางพฤติกรรม ส่วน MAE แยกกลุ่มยอดใช้จ่ายช่วยเปิดเผยว่าคะแนนเฉลี่ยซ่อนความผิดพลาดของผู้เล่นยอดสูงหรือไม่
''')
    code('''for task,m in metrics.items():
    importance=m.get('feature_importance')
    if not importance:
        continue
    values=pd.Series(importance).sort_values().tail(10)
    fig,ax=plt.subplots(figsize=(8,4))
    values.plot.barh(ax=ax,color='#236c9c')
    ax.set(title=f'Task {task}: top 10 CatBoost feature importance',xlabel='Relative model importance (%)',ylabel='Input feature')
    fig.tight_layout()
    plt.show()
path=OUTPUT_DIR/'task3_oof.csv'
if path.exists():
    errors=pd.read_csv(path)
    errors['absolute_error_thb']=(errors.target-errors.prediction).abs()
    errors['actual_spending_band']=pd.cut(errors.target,[-1e-9,0,1000,10000,np.inf],
                                          labels=['zero','0-1,000 THB','1,000-10,000 THB','10,000+ THB'])
    display(errors.groupby('actual_spending_band',observed=True).agg(
        players=('target','size'),mae_thb=('absolute_error_thb','mean')))
''')
    markdown('''## 7. สร้างไฟล์ส่งอย่างตรวจสอบได้

ต้องมีผลครบทั้ง 5 Task จึงสร้าง `submission.csv` โค้ดจะเช็กรหัสผู้เล่นภายในแต่ละ Task แล้วเรียงกลับตาม test.csv จากนั้นกำหนดให้แถวที่ i ตรงกับ ANS แถวที่ i โดยไม่ join ผู้เล่นข้าม Task ใช้ตัวคั่นตามไฟล์ตัวอย่างจริงและชื่อเกมจาก CSV

สมมติฐานลำดับ ANS, ชื่อเกมที่ตัวตรวจยอมรับ และข้อยกเว้น Task 5 ควรยืนยันกับผู้สอน การผ่านการตรวจในเครื่องยังไม่ใช่หลักฐานว่า Kaggle ยอมรับไฟล์หรือได้คะแนนเท่าใด Notebook นี้ไม่อัปโหลดคำตอบให้อัตโนมัติ
''')
    code('''missing = [t for t in range(1,6) if not (OUTPUT_DIR/f'task{t}_predictions.csv').exists()]
if missing:
    print('ยังไม่สร้างไฟล์ส่ง: ขาดผล Task',missing)
else:
    manifest=assemble(DATA_ROOT,OUTPUT_DIR,IMAGE_ROOT)
    print(validate_file(OUTPUT_DIR/'submission.csv',DATA_ROOT,IMAGE_ROOT))
    print('ไฟล์:',manifest['file'])
    display(pd.read_csv(OUTPUT_DIR/'submission.csv',sep=manifest['separator']).head())
''')
    markdown('''## 8. บันทึกสิ่งที่เรียนรู้และแผนทดลอง

สำหรับแต่ละ Task จด 1) ปัญหาและ metric 2) วิธีแบ่งข้อมูล 3) การจัดการค่าหาย/คลาส 4) โมเดลและเหตุผล 5) ผลเทียบ baseline 6) ข้อผิดพลาดและผลทางธุรกิจ 7) สิ่งที่ทดลองแล้วดีหรือแย่ลง

การทดลองที่มีประโยชน์ถัดไป: ปรับน้ำหนักคลาส Task 1–2 บน split เดิม, เปรียบเทียบโมเดลสองขั้นใน Task 3 ด้วย MAE จริง, ตรวจภาพใกล้ซ้ำและปรับ augmentation ใน Task 4, สำรวจความไวต่อ contamination ของ Task 5 โดยไม่อ้างว่าคะแนนดีขึ้นหากไม่มี label

อ่านคำอธิบายและตัวอย่างคำนวณเพิ่มเติมใน `STUDY_GUIDE_TH.md` และเก็บผลทดลองทุกครั้งแยกโฟลเดอร์ ไม่ใช้ public leaderboard เป็นชุด validation หลัก
''')
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python','version':'3.12'}}, 'cells':cells}
    for index,cell in enumerate(cells):cell['id']=f'pis-{index:03d}'
    path=ROOT/'Player_Intelligence_System.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(path)


if __name__=='__main__':main()
