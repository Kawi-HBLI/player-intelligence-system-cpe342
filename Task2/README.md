# Task 2: Player Segmentation

จำแนกผู้เล่นเป็น 4 กลุ่ม: Casual (0), Grinder (1), Social (2), Whale (3) วัดด้วย Macro F1

## ข้อมูล

- train 104,000 แถว มี label ครบ
- 34 ฟีเจอร์ตัวเลขและ 10 ฟีเจอร์หมวดหมู่ ค่าว่างเฉลี่ย 8.3%
- สัดส่วนคลาส 39% / 25% / 20% / 15%
- `random_metric_1` ถึง `random_metric_3` เป็น noise ตามที่ datacard ระบุ

## วิธีทำ

- CatBoost (MultiClass) ใช้ฟีเจอร์หมวดหมู่ตรงๆ ผ่าน `cat_features` ค่าว่างของหมวดหมู่แทนด้วย `__MISSING__`
- Stratified 3-fold แล้วเฉลี่ยความน่าจะเป็นบน test

## ผลในเครื่อง

Macro F1 (OOF) = 0.749 แยกตามคลาส: Casual 0.82, Grinder 0.74, Social 0.71, Whale 0.73

ทดลอง 5-fold เทรนนานขึ้นและตัด `random_metric` ออก ได้ 0.751 ต่างกันน้อยมาก โมเดลนี้จึงใกล้เพดานของข้อมูลแล้ว

## ไฟล์

- `train.py` รันงานนี้: `python Task2/train.py --folds 3`
- โค้ดหลักอยู่ที่ `pis/tabular.py`
- การทดลองเทรนนานขึ้น: `experiments/long_catboost_5fold.py`
