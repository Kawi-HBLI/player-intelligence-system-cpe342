# Player Intelligence System

โค้ดของทีม ML018 สำหรับการแข่ง Kaggle [player-intelligence-system-cpe-342-ml](https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml) วิชา CPE342 Machine Learning มี 5 งานจากข้อมูลเกมชุดเดียวกัน คะแนนรวมคือค่าเฉลี่ยของคะแนนทั้ง 5 งาน

## งานทั้ง 5 และโมเดลที่ใช้

| งาน | โจทย์ | ตัววัด | โมเดลสุดท้าย | คะแนนในเครื่อง |
|---|---|---|---|---|
| [1](Task1/README.md) | ตรวจจับคนโกง | F2 ของคลาสคนโกง | CatBoost 10 fold + ฟีเจอร์จากช่วงค่า, threshold 0.182 | OOF F2 0.8388 |
| [2](Task2/README.md) | แบ่งผู้เล่นเป็น 4 กลุ่ม | Macro F1 | naive Bayes ปรับเรียบ 0.6 ผสม CatBoost 0.4 | OOF 0.7567 |
| [3](Task3/README.md) | ทำนายยอดใช้จ่าย 30 วัน | 1 / (1 + NMAE) | V9a: classifier + CatBoost MultiQuantile และเติม `prev_month_spending` ที่ว่าง | OOF 0.7947 |
| [4](Task4/README.md) | จำแนกภาพหน้าจอ 5 เกม | Macro F1 | conv4p (ConvNeXt 3 สูตร + EfficientNetV2-S) 0.4 ผสม DINOv2 ViT-B/14 0.6, TTA 5 มุมมอง | scene OOF 0.9733 |
| [5](Task5/README.md) | หาบัญชีผิดปกติ ไม่มี label | F3 | detector v2 สามมุมมองแบบ Empirical Bayes | ไม่มี label |

รายละเอียดของแต่ละงาน (EDA, วิธีทำ, การทดลองที่ได้และไม่ได้ผล) อยู่ใน README ของงานนั้น

## โครงสร้าง

```text
run.py                    จุดเริ่มของ pipeline: audit, train, assemble, validate
pis/                      โค้ดหลักที่ใช้ร่วมกัน (tabular.py งาน 1-3, image.py, anomaly.py, submission.py)
Task1/ ... Task5/         train.py ของแต่ละงานและ README
Task4/final/              คำทำนายงาน 4 ของไฟล์ส่งสุดท้าย (conv4p 0.4 + DINOv2 0.6)
experiments/task1/        การทดลองและ EDA ของงาน 1
experiments/task2/        การทดลองและ EDA ของงาน 2
experiments/task3/        โมเดลสุดท้ายและการทดลองของงาน 3 (ดู LOG.md)
experiments/eda/          ตรวจสูตรตัววัดของงาน 2-5
tools/swap_submission.py  สลับคอลัมน์บางงานในไฟล์ส่ง
docs/KAGGLE_FINDINGS_TH.md  บันทึกสิ่งที่พบระหว่างการแข่ง
tests/                    unit tests
prepare_images.py         ดาวน์โหลดและแตกไฟล์ภาพของงาน 4
```

## ติดตั้ง

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-experiments.txt
```

`requirements-experiments.txt` ใช้กับสคริปต์ใน `experiments/` บางตัวเท่านั้น (LightGBM, XGBoost, Optuna) pipeline หลักกับโมเดลสุดท้ายของงาน 3 ใช้แค่ `requirements.txt`

ข้อมูลของการแข่งไม่อยู่ใน repo ให้วางโฟลเดอร์ `01_binary_classification` ถึง `05_anomaly_detection` และ `sample_submission.csv` ไว้ที่ `../cpe342-kaggle-main` (ข้างโฟลเดอร์ repo) หรือระบุที่อื่นด้วย `--data-root` ภาพของงาน 4 แจกแยก ดาวน์โหลดได้ด้วย `python prepare_images.py --download`

## ทำไฟล์ส่ง (submission.csv)

รันจากโฟลเดอร์หลักของ repo ตามลำดับนี้

```powershell
python Task1/train.py --threads 24
python Task2/train.py --folds 3 --threads 24
python Task5/train.py
python experiments/task3/model/v9_impute/v9_impute.py --threads 12
python experiments/task3/model/refit/refit.py --variant v9a --threads 12
cp outputs/experiments/task3_v9a_refit/task3_predictions.csv outputs/main/
cp Task4/final/task4_predictions.csv outputs/main/
python run.py assemble
```

ผลอยู่ที่ `outputs/main/submission.csv` และ `run.py assemble` ตรวจรูปแบบไฟล์ให้ก่อนเขียน

- CatBoost บน CPU สร้างต้นไม้ต่างกันเล็กน้อยเมื่อจำนวน thread ไม่เท่ากัน ถ้าต้องการผลตรงกับไฟล์ที่ส่งจริงให้ใช้จำนวน thread ตามข้างบน
- งาน 4 ไม่ต้องเทรนใหม่ คำทำนายที่ผสม conv4p 0.4 กับ DINOv2 0.6 แล้วเก็บไว้ใน `Task4/final/` คำสั่งสร้างใหม่จาก logits และคำสั่งเทรนทุก run อยู่ใน [Task4/README.md](Task4/README.md)
- `v9_impute.py` ใช้ราว 17 นาที `refit.py` ราว 6 นาที งานอื่นใช้ไม่กี่นาที

## เทสต์

```powershell
python -m unittest discover -s tests
```

## ข้อตกลงเรื่องข้อมูล

- TA ตอบไว้เมื่อ 15 พ.ย. ว่าให้ถือ test เป็นข้อมูลที่ไม่เคยเห็น ทุกงานทำ EDA และเลือกโมเดลจาก train อย่างเดียว ไม่ใช้สถิติของ test และไม่ทำ pseudo-label ส่วน test-time augmentation ทำได้
- งาน 5 มีแต่ `test.csv` จึง fit บนไฟล์นั้นตามโจทย์ โดยไม่ใช้ลำดับแถวหรือเลข `player_id`
- งาน 4 ใช้ weight ImageNet-1k ที่มากับ torchvision เท่านั้น
- ไฟล์ส่งจับคู่แต่ละงานกับ id ของ `sample_submission.csv` ตามลำดับแถวของ `test.csv` ของงานนั้น ยกเว้นงาน 4 ที่ `test.csv` ชุดแก้ไข (12 พ.ย.) ระบุ id ของทุกภาพไว้ คำตอบงาน 4 ส่งเป็นเลข 0-4 (7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4)
