# Task 3: Monthly Spending Prediction

ทำนายยอดใช้จ่ายใน 30 วันข้างหน้า หน่วยบาท วัดด้วย 1 / (1 + NMAE) โดย NMAE คือผลรวมค่าคลาดเคลื่อนสัมบูรณ์ หารด้วยผลรวมยอดจริง

## ข้อมูล

- train 104,000 แถว 32 ฟีเจอร์ ค่าว่างเฉลี่ย 5.5%
- เป้าหมายเบ้ขวามาก: 48% เป็น 0, median 79 บาท, สูงสุด 300,000 บาท

## โมเดลสุดท้าย: V9a

โค้ด ประวัติการทดลองทั้งหมด และกฎการเลือกโมเดลอยู่ใน [experiments/task3/](../experiments/task3/LOG.md) ส่วนนี้สรุปตัวที่ใช้ทำไฟล์ส่ง

- ขั้นแรก CatBoost classifier ทายความน่าจะเป็น p ที่ผู้เล่นจะใช้เงิน
- ขั้นที่สอง CatBoost MultiQuantile ทำนาย log ของยอดต่อยอดเดือนก่อนหลาย quantile แล้วใช้ quantile ที่ 1 − 0.5/p ค่าตั้งของขั้นนี้ได้จาก Optuna (V8)
- เติม `prev_month_spending` ที่ว่างด้วย HistGradientBoosting จากฟีเจอร์อื่นโดยไม่ใช้ label
- เทรนบน train ทั้งหมด 3 seed เฉลี่ยกัน แล้วตัดค่าไว้ในช่วง 0 ถึง 300,000 บาท

```powershell
python experiments/task3/model/v9_impute/v9_impute.py --threads 12
python experiments/task3/model/refit/refit.py --variant v9a --threads 12
```

คำทำนายอยู่ที่ `outputs/experiments/task3_v9a_refit/task3_predictions.csv`

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

- `train.py` รัน baseline เดิม (CatBoost MAE ผ่าน `pis/tabular.py`): `python Task3/train.py --folds 3` ใช้เทียบเท่านั้น ไม่ได้ใช้ทำไฟล์ส่ง
- โมเดลสุดท้ายและการทดลองทั้งหมดอยู่ใน `experiments/task3/model/`
- การทดลองเทรนนานขึ้นของ baseline: `experiments/long_catboost_5fold.py`
