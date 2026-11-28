# Task 3: Monthly Spending Prediction

ทำนายยอดใช้จ่ายใน 30 วันข้างหน้า หน่วยบาท วัดด้วย 1 / (1 + NMAE) โดย NMAE คือผลรวมค่าคลาดเคลื่อนสัมบูรณ์ หารด้วยผลรวมยอดจริง

## ข้อมูล

- train 104,000 แถว 32 ฟีเจอร์ ทุกฟีเจอร์มีค่าว่าง 5.7–6.0% (median 5.8%)
- เป้าหมายเบ้ขวามาก: 48% เป็น 0, median 79 บาท, สูงสุด 300,000 บาท

## โมเดลสุดท้าย: V9a

โค้ด ประวัติการทดลองทั้งหมด และกฎการเลือกโมเดลอยู่ใน [experiments/task3/](../experiments/task3/LOG.md) ส่วนนี้สรุปตัวที่ใช้ทำไฟล์ส่ง

- ขั้นแรก CatBoost classifier ทายความน่าจะเป็น p ที่ผู้เล่นจะใช้เงิน
- ขั้นที่สอง CatBoost MultiQuantile ทำนาย log ของยอดต่อยอดเดือนก่อนหลาย quantile แล้วใช้ quantile ที่ 1 − 0.5/p ค่าตั้งของขั้นนี้ได้จาก Optuna (V8)
- เติม `prev_month_spending` ที่ว่างด้วย HistGradientBoosting จากฟีเจอร์อื่นโดยไม่ใช้ label
- เทรนบน train ทั้งหมด 3 seed เฉลี่ยกัน แล้วตัดค่าไว้ในช่วง 0 ถึง 300,000 บาท

```powershell
python Task3/train.py --threads 12
```

`train.py` รันสองสคริปต์ใน `experiments/task3/model/` ต่อกัน คือ `v9_impute/v9_impute.py` (CV 3 fold ราว 17 นาที) แล้ว `refit/refit.py --variant v9a` (เทรนบน train ทั้งหมดราว 6 นาที) จากนั้นคัดลอกคำทำนายจาก `outputs/experiments/task3_v9a_refit/task3_predictions.csv` ไปที่ `outputs/main/` ให้ `python run.py assemble` ใช้ต่อ ถ้าเคยรัน CV แล้ว ใส่ `--skip-cv` เพื่อรันแค่ refit ได้ ใช้ 12 thread ถึงจะได้ผลตรงกับไฟล์ที่ส่ง

สองสคริปต์นี้เขียนผลทับ `v9_impute_results.md`, `refit_v9a_results.md` และแถวของตัวเองใน `experiments/task3/LOG.md` ถ้าผลเหมือนเดิมให้ `git checkout` ไฟล์เหล่านี้คืน

## ผลในเครื่อง

KFold 3 fold ชุดเดียวกับ baseline ตั้งแต่ V8 วัดแบบ honest early stopping (early stop บนส่วนที่แยกจาก train ของ fold นั้น) ซึ่ง V4 วัดแบบนี้ได้ 0.7911

| รุ่น | สิ่งที่เปลี่ยน | OOF |
|---|---|---|
| baseline | CatBoost MAE บนยอดเงิน | 0.7753 |
| V3 | แยกสองขั้น ขั้นที่สองทำนาย log ของยอดต่อยอดเดือนก่อน | 0.7900 |
| V4 | MultiQuantile และเลือก quantile ตาม p | 0.7918 |
| V8 | ตั้งค่าขั้นที่สองด้วย Optuna (honest) | 0.7921 |
| V9a | เติม `prev_month_spending` ที่ว่าง (honest) | 0.7947 |

## ไฟล์

- `train.py` รันโมเดลสุดท้าย V9a ตามข้างบน ส่วน `python Task3/train.py --baseline --folds 3` รัน baseline เดิม (CatBoost MAE ผ่าน `pis/tabular.py`) ใช้เทียบเท่านั้น ไม่ได้ใช้ทำไฟล์ส่ง
- โค้ดของโมเดลสุดท้ายและการทดลองทั้งหมดอยู่ใน `experiments/task3/model/`
- `baseline/task3_oof.csv` คือ OOF ของ baseline ที่สคริปต์ของงาน 3 ใช้เช็คความถูกต้องตอนเริ่ม
- การทดลองเทรนนานขึ้นของ baseline: `experiments/long_catboost_5fold.py`
