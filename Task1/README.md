# Task 1: Anti-Cheat Detection

จำแนกบัญชีที่ถูก flag ว่าเป็นคนโกงจริง (1) หรือผู้เล่นปกติ (0) วัดด้วย Macro F2

## ข้อมูล

- train 104,000 แถว มี label 90,816 แถว อีก 13,184 แถวไม่มี label จึงตัดออกจากการเทรน
- 31 ฟีเจอร์ตัวเลข ค่าว่างเฉลี่ย 11.6% ต่อคอลัมน์
- คนโกงคิดเป็น 35% ของแถวที่มี label

## วิธีทำ

- CatBoost (Logloss, depth 6, learning rate 0.05, สูงสุด 1,000 รอบ, early stopping 80 รอบ) ส่งค่าว่างเข้าโมเดลตรงๆ ไม่ impute
- Stratified 3-fold แล้วเฉลี่ยความน่าจะเป็นของ 3 โมเดลบน test
- F2 ให้น้ำหนัก recall มากกว่า precision จึงเลือก threshold จากค่า OOF ให้ macro F2 สูงสุด ได้ 0.385 แทน 0.5

## ผลในเครื่อง

| การตั้งค่า | Macro F2 (OOF) |
|---|---|
| threshold 0.5 | 0.758 |
| threshold 0.385 | 0.777 |
| 5-fold เทรนนานขึ้น threshold 0.39 | 0.777 |

คะแนนที่ threshold 0.385 สูงกว่าความจริงเล็กน้อย เพราะใช้ OOF ชุดเดียวกันทั้งเลือก threshold และวัดผล

## ไฟล์

- `train.py` รันงานนี้: `python Task1/train.py --folds 3`
- โค้ดหลักอยู่ที่ `pis/tabular.py` (ใช้ร่วมกับ Task 2 และ 3)
- การทดลองเทรนนานขึ้น: `experiments/long_catboost_5fold.py`
