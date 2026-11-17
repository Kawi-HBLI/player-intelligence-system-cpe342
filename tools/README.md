# tools/

## swap_submission.py — สลับคอลัมน์ task เข้าไฟล์ submission ที่ดีที่สุดของทีม

ใช้เมื่อปรับ task ของตัวเองแล้ว อยากเอาเฉพาะคอลัมน์นั้นไปแทนในไฟล์ Kaggle ที่ดีที่สุด
โดยคอลัมน์อื่นไม่เปลี่ยนเลย (เช็คแบบ byte-for-byte ให้หลังเขียนไฟล์)

**แบบที่ 1 — คัดลอกคอลัมน์จาก submission เต็มอีกไฟล์** (จับคู่แถวด้วย `id`)

```bash
python tools/swap_submission.py --base best.csv --from my_submission.csv --tasks task4
```

**แบบที่ 2 — ใช้ไฟล์ prediction ราย task** (`player_id,prediction` เหมือน `outputs/main/taskN_predictions.csv`)

```bash
python tools/swap_submission.py --base best.csv --pred task3=outputs/main/task3_predictions.csv
```

ใช้สองแบบพร้อมกันได้ และใส่ `--pred` ได้หลายครั้ง เช่น
`--from a.csv --tasks task4 --pred task3=t3.csv --pred task5=t5.csv`

- แบบที่ 1 จะแจ้งด้วยว่าไฟล์ `--from` ต่างจาก base ในคอลัมน์อื่นกี่แถว (แค่แจ้ง ไม่คัดลอก เช่นเพื่อนสร้างไฟล์จาก base เก่า)
- ผลลัพธ์ default: `outputs/kaggle/<ชื่อ base>__<task ที่สลับ>.csv` (เปลี่ยนด้วย `--out`) ไม่เขียนทับ `--base`
- แบบที่ 2 ต้องมีข้อมูล Kaggle (`--data-root`, default `../cpe342-kaggle-main`); task4 ต้องมี `test.csv` ของรูป (`--image-root` ถ้าอยู่ที่อื่น)
- task4 ต้องเป็นตัวเลข 0–4 เหมือนไฟล์ base (ส่งเป็นข้อความแล้วคะแนน Kaggle ต่ำกว่า)
- สุดท้ายรัน validator ของ repo ให้ ถ้าไม่ผ่านจะลบไฟล์ผลลัพธ์และ exit code ≠ 0
