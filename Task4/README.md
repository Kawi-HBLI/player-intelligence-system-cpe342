# Task 4: Game Title Detection

จำแนกภาพ screenshot ว่ามาจากเกมไหนใน 5 เกม วัดด้วย Macro F1

## ข้อมูล

- ภาพ train 31,546 ภาพ และ test 25,889 ภาพ ขนาด 230x120 ดาวน์โหลดแยกจาก Google Drive ตามลิงก์ใน README ของข้อมูล
- ภาพถูกหมุน ใส่ noise และบิดสีมาแล้วทั้ง train และ test
- คลาสไม่สมดุล: 7-11k 12,429 / pube 5,926 / freefried 5,596 / fiveN 4,407 / roblock 3,188

## วิธีทำ (ตัวหลัก baseline, `pis/image.py`)

- ResNet18 ที่ pretrain บน ImageNet (กติกาอนุญาตเฉพาะงานนี้) fine-tune 5 epoch ด้วย AdamW
- ปรับขนาดภาพเป็น 128x256 และใช้ ColorJitter ตอนเทรน
- แบ่ง holdout 20% แบบ stratified และจัดภาพที่ไบต์ซ้ำกัน (SHA-256) ไว้ฝั่งเดียวกัน กันภาพซ้ำรั่วข้ามฝั่ง
- label เป็นเลขตาม datacard: 7-11k=0, fiveN=1, freefried=2, pube=3, roblock=4 (`train.csv` ชุดใหม่ใช้เลขชุดนี้อยู่แล้ว)

Macro F1 บน holdout = 0.913 แยกตามคลาส: 7-11k 0.96, fiveN 0.91, freefried 0.91, pube 0.91, roblock 0.87

## วิธีทำ (ตัวใหม่, `Task4/train.py`)

- ConvNeXt-Tiny (น้ำหนัก torchvision `IMAGENET1K_V1` เท่านั้น) 15 epoch, cosine, lr backbone 2e-4 / head 3e-4
- augmentation แรงบน GPU: หมุน ±30° ซูม มืด/สว่าง contrast saturation สีเพี้ยน (hue) เบลอ noise
- แบ่ง 5 fold แบบ StratifiedGroupKFold โดยจับภาพที่เป็นเฟรมเดียวกัน (near-copy) ไว้ fold เดียวกัน ทดลองบน fold 0
- ทุกการตัดสินใจใช้แค่ภาพ train ภาพ test เปิดแค่ตอน `--mode predict`

Macro F1 บน fold 0 = 0.9896 แยกตามคลาส: 7-11k 0.996, fiveN 0.990, freefried 0.992, pube 0.990, roblock 0.980

## ปัญหา test.csv ของผู้จัด

ก่อนวันที่ 12 พ.ย. คะแนนงานนี้บน leaderboard อยู่ระดับเดาสุ่ม ทั้งที่ holdout ได้ 0.913 เราตรวจแล้วว่าไฟล์ตรงกับต้นทาง โมเดลอิสระอีกตัวทำนายตรงกับโมเดลหลัก 63% บน test และลำดับแถวสองแบบให้ผลไม่ต่างกัน จึงสรุปว่าปัญหาไม่ได้อยู่ฝั่งเรา

ผู้จัดยืนยันและแก้ `test.csv` ไฟล์ใหม่มีคอลัมน์ `id` (รหัส ANS), `file_name` และ `label` ภาพไม่เปลี่ยน จึงไม่ต้องเทรนใหม่ ตอนรวม submission โค้ดจับคู่คำทำนายกับรหัส ANS ผ่าน `file_name`

## ไฟล์

- `train.py` ตัวใหม่ ต้องใช้ PyTorch ที่รองรับ CUDA (ราว 1 ชม. บน RTX 4050):
  - เทรน: `python Task4/train.py --run-name r3d_lr2e-4 --model convnext_tiny --split near-copy --epochs 15 --schedule cosine --aug strong --lr-backbone 2e-4`
  - ทายภาพ test: `python Task4/train.py --mode predict --run-name r3d_lr2e-4` ได้ `outputs/main/task4_predictions.csv` สำหรับ `python run.py assemble` และ `outputs/task4_exp/<run>/submission.csv` (= `baseline/submission.csv` ที่เปลี่ยนแค่คอลัมน์ task4)
  - ผลแต่ละรอบอยู่ใน `outputs/task4_exp/` (`results.csv` รวมทุกรอบ) หยุดกลางคันแล้วรันคำสั่งเดิม + `--resume` เพื่อทำต่อ
- ตัวหลักเดิม (ResNet18) ยังรันได้ด้วย `python run.py train --tasks 4 --epochs 5 --batch-size 128` โค้ดอยู่ที่ `pis/image.py`
- เตรียมภาพ: `python prepare_images.py --download`
