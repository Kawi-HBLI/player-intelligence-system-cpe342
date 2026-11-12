# Task 4: Game Title Detection

จำแนกภาพ screenshot ว่ามาจากเกมไหนใน 5 เกม วัดด้วย Macro F1

## ข้อมูล

- ภาพ train 31,546 ภาพ และ test 25,889 ภาพ ขนาด 230x120 ดาวน์โหลดแยกจาก Google Drive ตามลิงก์ใน README ของข้อมูล
- ภาพถูกหมุน ใส่ noise และบิดสีมาแล้วทั้ง train และ test
- คลาสไม่สมดุล: 7-11k 12,429 / pube 5,926 / freefried 5,596 / fiveN 4,407 / roblock 3,188

## วิธีทำ

- ResNet18 ที่ pretrain บน ImageNet (กติกาอนุญาตเฉพาะงานนี้) fine-tune 5 epoch ด้วย AdamW
- ปรับขนาดภาพเป็น 128x256 และใช้ ColorJitter ตอนเทรน
- แบ่ง holdout 20% แบบ stratified และจัดภาพที่ไบต์ซ้ำกัน (SHA-256) ไว้ฝั่งเดียวกัน กันภาพซ้ำรั่วข้ามฝั่ง
- label เป็นเลขตาม datacard: 7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4 (`train.csv` ชุดใหม่ใช้เลขชุดนี้อยู่แล้ว)

## ผลในเครื่อง

Macro F1 บน holdout = 0.913 แยกตามคลาส: 7-11k 0.96, fiveN 0.91, freefried 0.91, pube 0.91, roblock 0.87

## ปัญหา test.csv ของผู้จัด

ก่อนวันที่ 12 พ.ย. คะแนนงานนี้บน leaderboard อยู่ระดับเดาสุ่ม ทั้งที่ holdout ได้ 0.913 เราตรวจแล้วว่าไฟล์ตรงกับต้นทาง โมเดลอิสระอีกตัวทำนายตรงกับโมเดลหลัก 63% บน test และลำดับแถวสองแบบให้ผลไม่ต่างกัน จึงสรุปว่าปัญหาไม่ได้อยู่ฝั่งเรา

ผู้จัดยืนยันและแก้ `test.csv` ไฟล์ใหม่มีคอลัมน์ `id` (รหัส ANS), `file_name` และ `label` ภาพไม่เปลี่ยน จึงไม่ต้องเทรนใหม่ ตอนรวม submission โค้ดจับคู่คำทำนายกับรหัส ANS ผ่าน `file_name`

## ไฟล์

- `train.py` รันงานนี้: `python Task4/train.py --epochs 5 --batch-size 128`
- โค้ดหลักอยู่ที่ `pis/image.py`
- เตรียมภาพ: `python prepare_images.py --download`
