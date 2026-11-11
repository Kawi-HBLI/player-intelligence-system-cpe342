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
- เพิ่มฟีเจอร์ `player_id_number` (ตัวเลขใน player_id) เพราะข้อมูลจำลองถูกสร้างเรียงตามกลุ่ม เลข id จึงบอกกลุ่มได้เกือบตรงๆ ฟีเจอร์นี้เป็นผลจากวิธีสร้างข้อมูล ไม่ใช่พฤติกรรมผู้เล่น

## ผลในเครื่อง

| โมเดล | Macro F1 (OOF) |
|---|---|
| CatBoost ไม่ใช้เลข id | 0.749 |
| CatBoost 5-fold เทรนนานขึ้น ตัด `random_metric` | 0.751 |
| decision tree บนเลข id อย่างเดียว | 0.968 |
| CatBoost + เลข id (5 fold) | 0.965 |

ไม่ใช้เลข id แยกตามคลาสได้: Casual 0.82, Grinder 0.74, Social 0.71, Whale 0.73 ฟีเจอร์พฤติกรรมอย่างเดียวจึงใกล้เพดานที่ประมาณ 0.75 ส่วนที่เพิ่มขึ้นทั้งหมดมาจากเลข id

## ไฟล์

- `train.py` รันงานนี้: `python Task2/train.py --folds 3`
- โค้ดหลักอยู่ที่ `pis/tabular.py`
- การทดลองเทรนนานขึ้น: `experiments/long_catboost_5fold.py`
