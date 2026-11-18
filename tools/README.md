# tools/

## swap_submission.py — สลับคอลัมน์ task เข้าไฟล์ submission ที่ดีที่สุดของทีม

ใช้เมื่อปรับ task ของตัวเองแล้ว อยากเอาเฉพาะคอลัมน์นั้นไปแทนในไฟล์ Kaggle ที่ดีที่สุด
โดยคอลัมน์อื่นไม่เปลี่ยนเลย (เช็คแบบ byte-for-byte ให้หลังเขียนไฟล์)

> path ในตัวอย่างเป็นแค่ตัวอย่าง ให้เปลี่ยนเป็น path ไฟล์ของตัวเอง

**แบบที่ 1 — คัดลอกคอลัมน์จาก submission เต็มอีกไฟล์** (จับคู่แถวด้วย `id`)

```bash
python tools/swap_submission.py --base best.csv --from my_submission.csv --tasks task4
```

**แบบที่ 2 — ใช้ไฟล์ prediction ราย task** (`player_id,prediction` เหมือน `outputs/main/taskN_predictions.csv`)

```bash
python tools/swap_submission.py --base best.csv --pred task3=outputs/main/task3_predictions.csv
python tools/swap_submission.py --base best.csv --pred task4=task4_predictions.csv --image-root Task4
```

ใช้สองแบบพร้อมกันได้ และใส่ `--pred` ได้หลายครั้ง เช่น
`--from a.csv --tasks task4 --pred task3=t3.csv --pred task5=t5.csv`

**ต้องใช้ `--data-root` / `--image-root` เมื่อไหร่**
- `--data-root`: โฟลเดอร์ข้อมูล Kaggle (default `../cpe342-kaggle-main`) ต้องมีสำหรับแบบที่ 2 เสมอ (ใช้ `test.csv` ของแต่ละ task) ใส่เองถ้าเก็บไว้ที่อื่น
- `--image-root`: ใช้กับ `--pred task4=...` เท่านั้น ชี้ไปที่โฟลเดอร์ที่มี `test.csv` ของรูป (เช่น `--image-root Task4`) ไม่ต้องมี `train.csv`
- แบบที่ 1 ใช้ได้แม้ไม่มีโฟลเดอร์ข้อมูล: จะใช้ id จากไฟล์ base แทน ข้าม validator ของ repo แต่ยังเช็คของ tool เอง (คอลัมน์, id, label, task3 ≥ 0)

**อื่น ๆ**
- แบบที่ 1 จะแจ้งด้วยว่าไฟล์ `--from` ต่างจาก base ในคอลัมน์อื่นกี่แถว (แค่แจ้ง ไม่คัดลอก เช่นเพื่อนสร้างไฟล์จาก base เก่า)
- label ของ task1/2/4/5 จะถูกเขียนเป็นจำนวนเต็มเสมอ (`3.0` → `3`); task4 ต้องเป็นตัวเลข 0–4 (ส่งเป็นข้อความแล้วคะแนน Kaggle ต่ำกว่า)
- ผลลัพธ์ default: `outputs/kaggle/<ชื่อ base>__<task ที่สลับ>.csv` (เปลี่ยนด้วย `--out`) ไม่เขียนทับ `--base`
- เขียนลงไฟล์ชั่วคราวก่อน เช็คผ่านแล้วค่อยแทนที่ `--out` ถ้าไม่ผ่าน exit code ≠ 0 และไฟล์ `--out` เดิมไม่ถูกแตะ

**⚠️ สัดส่วน class ที่ tool พิมพ์ออกมา (before/after) ใช้เช็คความสมเหตุสมผลเท่านั้น**
ห้ามเอาสัดส่วนจาก prediction ของ test ไปปรับโมเดล (class weights, thresholds, priors) เด็ดขาด
TA นับว่าเป็น pseudo-labeling
