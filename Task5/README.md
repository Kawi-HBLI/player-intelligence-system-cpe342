# Task 5: Account Security Monitoring

ตรวจหาบัญชีที่มีพฤติกรรมผิดปกติ (1) แบบ unsupervised วัดด้วย Macro F3

## ข้อมูล

- มีเฉพาะ `test.csv` 25,889 แถว ไม่มี train และไม่มี label
- 122 ฟีเจอร์ ส่วนใหญ่วัดซ้ำ 4 ช่วงเวลา (`_1` ถึง `_4`) และมี security flag 4 ตัว
- ประมาณ 3.3% ของแถวมีค่ากระโดดเกิน 10 MAD ส่วนใหญ่เกิดในช่วงที่ 4 เช่น ตำแหน่ง login, skill_trajectory, currency_transfers

## วิธีทำ

- สร้างฟีเจอร์การเปลี่ยนแปลงตามเวลา: ส่วนต่างของช่วงที่ 4 กับค่าเฉลี่ย 3 ช่วงแรก, range, ระยะทางระหว่างตำแหน่ง login
- Isolation Forest สองตัว ตัวหนึ่งบนฟีเจอร์ดิบ อีกตัวบนฟีเจอร์ตามเวลา
- เฉลี่ย percentile rank ของสองคะแนน แล้ว flag 20% ที่คะแนนสูงสุด
- shift score (robust z ของการกระโดดในแต่ละช่วง บวก security flag) คำนวณเก็บไว้เทียบ แต่ไม่ได้ใช้ตัดสิน

งานนี้ fit กับ `test.csv` ที่ไม่มี label เพราะโจทย์ให้ข้อมูลมาแค่ไฟล์นี้ ไม่ได้ใช้ label หรือ pseudo-label ของ test

## การตรวจผลโดยไม่มี label

`synthetic_eval.py` ฉีด anomaly เทียมแบบเดียวกับที่พบในข้อมูลจริงเข้าไปตามสัดส่วนที่กำหนด แล้ววัด macro F3 ของ detector แต่ละแบบ

| สัดส่วนที่ฉีด | IF raw+temporal (เดิม) | IF temporal + shift score |
|---|---|---|
| 5% | 0.873 | 0.930 |
| 10% | 0.892 | 0.948 |
| 15% | 0.884 | 0.955 |

ผลบน leaderboard ไม่เป็นไปตามตารางนี้ รอบส่ง H ใช้ IF temporal + shift score แล้วคะแนนรวมเพิ่มน้อยกว่าที่งาน 2 ควรให้ประมาณ 0.023 จึงน่าจะแย่กว่า detector เดิมบนเฉลยจริง โค้ดจึงกลับไปใช้ IF raw+temporal การตรวจด้วย anomaly เทียมบอกได้แค่ว่า detector จับ anomaly แบบที่เราฉีดได้ดีแค่ไหน

## ไฟล์

- `train.py` รันงานนี้: `python Task5/train.py --contamination 0.20`
- `synthetic_eval.py` ตรวจด้วย anomaly เทียม: `python Task5/synthetic_eval.py ../cpe342-kaggle-main`
- โค้ดหลักอยู่ที่ `pis/anomaly.py`
