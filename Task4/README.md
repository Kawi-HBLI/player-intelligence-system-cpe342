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

## รอบปรับปรุงหลังเห็นคะแนน Kaggle (branch `task4-boost`)

ens18 (ConvNeXt-Tiny 18 ตัว) ได้ราว 0.99 บน near-copy fold แต่คะแนนบน Kaggle ต่ำกว่านั้นมาก ช่องว่างขนาดนี้บอกว่า validation แบบ near-copy ยังง่ายเกินไป ภาพที่ไม่ใช่เฟรมเดียวกันแต่มาจากฉากเดียวกันยังกระจายอยู่ทั้งฝั่ง train และ validation รอบนี้จึงเปลี่ยน validation ก่อน แล้วค่อยลองสูตรเทรน

### validation แบบ scene (`--split scene`)

- ดึง feature ของทุกภาพด้วย ResNet18 (ImageNet) แล้วแบ่งแต่ละเกมเป็น 20 กลุ่มด้วย k-means ภาพกลุ่มเดียวกันต้องอยู่ fold เดียวกัน แบ่ง 5 fold ด้วย StratifiedGroupKFold seed 342
- feature จาก GPU ต่างกันได้ในหลักท้ายๆ ระหว่างรอบ กลุ่มจึงถูกเก็บไว้ใน `outputs/task4_exp/cache/groups_scene_seed342_n31546.json` ทุก run ของ split นี้จะถือภาพ validation ชุดเดียวกันพอดี ห้ามลบไฟล์นี้ถ้าจะเทียบกับ run เดิม
- ใช้น้ำหนัก epoch สุดท้าย (`final.pt`) ไม่ใช่ `best.pt` เพราะการเลือก epoch ด้วย validation ชุดเดียวกันทำให้ OOF สูงเกินจริงราว 0.005
- คะแนน scene (ราว 0.96) ยังสูงกว่าคะแนนบน Kaggle มาก ทิศทางตรงกัน แต่ขนาดไม่ตรง สูตรที่ชนะบน scene เพียงเล็กน้อยชนะบน Kaggle มากกว่านั้นหลายเท่า โดยเฉพาะตอนเพิ่มโมเดลคนละตระกูลเข้าไปในค่าเฉลี่ย ส่วนต่างระดับ 0.001-0.003 บน scene จึงใช้จัดอันดับสูตรไม่ได้

### ผล OOF 5 fold บน scene split

ภาพ train ครบ 31,546 ภาพ ทุกภาพถูกทายโดยโมเดลที่ไม่เคยเห็นมันตอนเทรน TTA rot8+zoom คือเฉลี่ย softmax ของภาพเดิม หมุน ±8° และซูม 1.12 กับ 0.9 (ชุด `TTA_VIEWS` ใน `train.py` เลือกจาก scene fold 0)

| ชุด | TTA | OOF | ผิด (ภาพ) | fold 0 / 1 / 2 / 3 / 4 |
|---|---|---|---|---|
| ConvNeXt-T aug strong (สูตร ens18) | ไม่มี | 0.9639 | 1019 | 0.9635 / 0.8897 / 0.9673 / 0.9861 / 0.9702 |
| ConvNeXt-T aug strong | rot8+zoom | 0.9650 | 990 | 0.9677 / 0.8907 / 0.9678 / 0.9866 / 0.9728 |
| ConvNeXt-T aug robust + multi-scale | rot8+zoom | 0.9636 | 1012 | 0.9680 / 0.8879 / 0.9697 / 0.9845 / 0.9709 |
| ConvNeXt-S aug robust + multi-scale | rot8+zoom | 0.9672 | 927 | 0.9689 / 0.8984 / 0.9749 / 0.9846 / 0.9744 |
| conv3 = สามชุดบนเฉลี่ยกัน | rot8+zoom | 0.9687 | 889 | 0.9730 / 0.8953 / 0.9755 / 0.9871 / 0.9771 |
| EfficientNetV2-S aug robust + multi-scale | rot8+zoom | 0.9660 | 954 | 0.9656 / 0.8909 / 0.9714 / 0.9866 / 0.9766 |
| conv4 = conv3 + EfficientNetV2-S (4 ชุด น้ำหนักเท่ากัน) | rot8+zoom | 0.9697 | 859 | 0.9732 / 0.8968 / 0.9750 / 0.9895 / 0.9792 |
| **conv4p** = conv3 + EfficientNetV2-S เฉพาะ fold 0-1 (ไฟล์ส่งสุดท้าย) | rot8+zoom | 0.9691 | 876 | 0.9732 / 0.8968 / 0.9755 / 0.9871 / 0.9771 |

fold 1 ยากกว่า fold อื่นชัดเจนในทุกสูตร คลาสที่ยังแย่สุดคือ roblock (F1 0.946 ใน conv3, 0.947 ใน conv4) และ fiveN (0.959 และ 0.961) OOF ของ conv4p ใน fold 0-1 เฉลี่ย 4 ชุด ส่วน fold 2-4 ไม่มี EfficientNetV2-S จึงเท่ากับ conv3

### ไฟล์ส่งสุดท้าย (conv4p)

เฉลี่ย softmax ของ 4 ชุดน้ำหนักเท่ากัน แล้วเลือกคลาสที่ความน่าจะเป็นสูงสุด แต่ละ run เฉลี่ย 5 มุมมอง (rot8+zoom) ก่อน แล้วเฉลี่ยข้าม fold ภายในชุด

| ชุด | run | fold |
|---|---|---|
| ConvNeXt-T aug strong | `cnt_sc_f{fold}` | 0-4 |
| ConvNeXt-T aug robust + multi-scale | `cntRS_sc_f{fold}` | 0-4 |
| ConvNeXt-S aug robust + multi-scale | `cnsRS2_sc_f{fold}` | 0-4 |
| EfficientNetV2-S aug robust + multi-scale | `ev2sRS_sc_f{fold}` | 0-1 |

EfficientNetV2-S fold 2-4 เทรนเสร็จทีหลัง ทีมลองทั้งแบบ 2 fold และ 5 fold แล้วเลือก conv4p ตามผลบน Kaggle ทั้งที่ conv4 ได้ OOF สูงกว่าเล็กน้อย ซึ่งตรงกับข้อสังเกตข้างบนว่า scene CV จัดอันดับส่วนต่างเล็กๆ ไม่ได้

- คำทำนายของไฟล์นี้อยู่ใน `Task4/final/task4_predictions.csv` (และความน่าจะเป็นใน `task4_predictions.probs.npy`) เก็บไว้ใน repo เพราะ logits ของ 17 run อยู่ใน `outputs/` ซึ่งไม่ขึ้น git ถ้าจะรวมไฟล์ส่ง ให้คัดลอก csv ไปที่ `outputs/main/` แล้วรัน `python run.py assemble`
- ถ้ามี run ครบในเครื่อง สร้างใหม่ได้ด้วยคำสั่งนี้ ผลตรงกับไฟล์ใน `final/` ทุกแถว:

```bash
python Task4/oof.py --group "cnt=cnt_sc_f{fold}" --group "cntRS=cntRS_sc_f{fold}" --group "cnsRS2=cnsRS2_sc_f{fold}" --group "ev2sRS=ev2sRS_sc_f{fold}@0,1" --views rot8+zoom --write-pred Task4/final/task4_predictions.csv
```

- ถ้าจะเทรนใหม่ทั้งหมด ใช้ 17 run (ราว 1 ถึง 1.5 ชั่วโมงต่อ run บน RTX 4050) ทุก run ใช้ `--split scene --epochs 15 --schedule cosine` แล้วตามด้วย `python Task4/train.py --mode tta --run-name <run>` ส่วนที่ต่างกันแต่ละชุดคือ
  - `cnt`: `--model convnext_tiny --micro-batch 64 --aug strong --lr-backbone 2e-4`
  - `cntRS`: `--model convnext_tiny --micro-batch 64 --aug robust --multi-scale --lr-backbone 2e-4`
  - `cnsRS2`: `--model convnext_small --micro-batch 32 --aug robust --multi-scale --lr-backbone 1.5e-4`
  - `ev2sRS`: `--model efficientnet_v2_s --micro-batch 64 --aug robust --multi-scale --lr-backbone 5e-4 --lr-head 1e-3`

### อะไรช่วย อะไรไม่ช่วย

ช่วย (อ้างอิง OOF 5 fold ข้างบน ส่วน conv3 ยืนยันด้วย Kaggle แล้ว)

- ConvNeXt-S ดีกว่า ConvNeXt-T ทุกแบบ และการเฉลี่ยหลายสูตรดีกว่าโมเดลเดี่ยวที่ดีที่สุด 38 ภาพ
- EfficientNetV2-S เดี่ยวๆ แพ้ ConvNeXt ทุกตัว (0.9660) แต่พอใส่เป็นชุดที่ 4 ในค่าเฉลี่ย ภาพผิดลดจาก 889 เหลือ 859 ดีขึ้น 4 ใน 5 fold (fold 2 แย่ลง 2 ภาพ) น่าจะเพราะเป็นโครงสร้างคนละแบบกับ ConvNeXt จึงพลาดคนละภาพ บน Kaggle การเพิ่ม EfficientNetV2-S ก็ช่วยเช่นกัน ส่วนการเลือกแค่ 3 ใน 4 ชุด (เช่น cnt+cnsRS+ev2s ได้ 0.9702) ต่างกันน้อยเกินกว่าจะเชื่อได้ จึงใช้ครบ 4 ชุดน้ำหนักเท่ากัน
- TTA rot8+zoom ช่วย ConvNeXt-T สูตรเดิมนิดหน่อย (ผิดลด 29 ภาพ)

ไม่ช่วย

- เทรนแบบ multi-scale กับ aug robust ใน ConvNeXt-T ตัวเดียวไม่ดีกว่าสูตรเดิม ผลบวกที่เคยเห็นบน fold 0 มาจากการเลือก checkpoint เป็นส่วนใหญ่
- มุมมอง TTA ที่ขยายขนาดภาพ (size1.125, size1.25) ไม่อยู่ในชุดที่เลือกจาก fold 0
- มุมมอง TTA ที่ยืด contrast รายภาพ (`auto`) เพื่อช่วยภาพมืด: conv3 ได้ 0.9685 เทียบ 0.9687 เลยตัดทิ้ง ทั้งที่จาก OOF ภาพมืดหรือ contrast ต่ำ (ค่า std ของพิกเซลต่ำกว่า 31) ผิด 16.7% และคิดเป็น 55% ของภาพที่ผิดทั้งหมด ปัญหานี้ยังไม่มีวิธีแก้ที่ได้ผล

### ทดสอบภาพเสีย (`stress.py`)

เอาภาพ validation ของแต่ละ scene fold มาทำให้เสีย 15 แบบ เช่น JPEG คุณภาพต่ำ ลดความละเอียด เบลอ noise มืด contrast ต่ำ หมุน crop ยืด motion blur หมอก vignette และบังภาพบางส่วน แล้ววัด macro F1 แบบไม่ใช้ TTA ตัวเลขข้างล่างเป็นค่าเฉลี่ยของ 5 fold ผลนี้ไม่ได้ใช้เลือกสูตร conv4 เพราะสูตรถูกกำหนดไว้ก่อนแล้ว

| ชุด | ภาพปกติ | เฉลี่ย 15 แบบ | 5 แบบที่ไม่มีใน aug | vignette | หมอก | crop1.5 |
|---|---|---|---|---|---|---|
| ConvNeXt-T aug strong | 0.9554 | 0.9360 | 0.9284 | 0.9022 | 0.9174 | 0.8896 |
| ConvNeXt-T robust + multi-scale | 0.9564 | 0.9386 | 0.9231 | 0.8699 | 0.9107 | 0.9290 |
| ConvNeXt-S robust + multi-scale | 0.9582 | 0.9433 | 0.9307 | 0.8958 | 0.9137 | 0.9337 |
| EfficientNetV2-S robust + multi-scale | 0.9577 | 0.9378 | 0.9173 | 0.8569 | 0.8918 | 0.9297 |
| 3 ชุด robust (ไม่มี ConvNeXt-T strong) | 0.9618 | 0.9479 | 0.9336 | 0.8832 | 0.9217 | 0.9385 |
| conv4 ครบ 4 ชุด | 0.9621 | 0.9485 | 0.9359 | 0.8912 | 0.9260 | 0.9351 |

5 แบบที่ไม่มีใน aug ของสูตรไหนเลยคือ motion blur, posterize, หมอก, vignette และ sharpen

- conv4 ทนภาพเสียดีที่สุดในภาพรวม คะแนนเฉลี่ยตกจากภาพปกติราว 0.014
- จุดอ่อนใหญ่สุดคือ vignette (ขอบภาพมืด) กับหมอก EfficientNetV2-S แย่สุดในสองแบบนี้ ส่วน ConvNeXt-T aug strong ทนกว่าตัวอื่น การเก็บมันไว้ใน conv4 จึงมีประโยชน์ เทียบกับ 3 ชุด robust แล้ว vignette ดีขึ้น 0.008 หมอกดีขึ้น 0.004 แลกกับ crop1.5 ที่แย่ลง 0.003
- ภาพที่ทำให้มืดเอง (0.959) หรือลด contrast เอง (0.957) แทบไม่ทำให้คะแนนตก ต่างจากภาพมืดจริงใน OOF ที่ผิดบ่อยมาก ความยากของภาพมืดจริงจึงน่าจะมาจากเนื้อหาของภาพ (เช่น ฉากที่หน้าตาคล้ายกันข้ามเกม) มากกว่าความสว่าง ข้อนี้เป็นการตีความ ยังไม่ได้พิสูจน์

### ข้อจำกัดที่ควรรู้

- ติดตั้ง timm ในเครื่องนี้ไม่ได้ จึงใช้แค่น้ำหนัก ImageNet-1k ของ torchvision (ConvNeXt-T/S, EfficientNetV2-S) ไม่มีน้ำหนัก ImageNet-22k
- ไม่ใช้ test ช่วยตัดสินใจเรื่องใดเลยตามคำตัดสินของ TA วันที่ 15 พ.ย. ไม่ดูภาพ ไม่ดูค่าสถิติ ไม่ดูสัดส่วนคำทาย ภาพ test ถูกเปิดแค่ตอนทำนายและตอนทำ TTA
- label ใน `train.csv` ชุดใหม่อ่านมาเป็น int32 ต้องแปลงเป็น long ก่อนเข้า loss ซึ่ง `train.py` ทำให้แล้ว
- EfficientNetV2-S (`--model efficientnet_v2_s --lr-backbone 5e-4 --lr-head 1e-3 --aug robust --multi-scale`) fold 0-1 เทรนด้วย `--micro-batch 64` แต่ fold 2-4 ต้องลดเป็น 32 เพราะที่ 64 หน่วยความจำ GPU ล้นไปใช้ RAM ของเครื่อง epoch หนึ่งช้าเป็นชั่วโมง batch ยังเป็น 128 เท่าเดิม แต่ BatchNorm ของ EfficientNet เห็นภาพทีละ 32 แทน 64 สูตรของ 5 fold จึงไม่เหมือนกันเป๊ะ (ConvNeXt ใช้ LayerNorm เลยไม่มีผลนี้)

## ไฟล์

- `train.py` ตัวใหม่ ต้องใช้ PyTorch ที่รองรับ CUDA (ราว 1 ชม. บน RTX 4050):
  - เทรน: `python Task4/train.py --run-name r3d_lr2e-4 --model convnext_tiny --split near-copy --epochs 15 --schedule cosine --aug strong --lr-backbone 2e-4`
  - ทายภาพ test: `python Task4/train.py --mode predict --run-name r3d_lr2e-4` ได้ `outputs/main/task4_predictions.csv` สำหรับ `python run.py assemble`
  - ผลแต่ละรอบอยู่ใน `outputs/task4_exp/` (`results.csv` รวมทุกรอบ) หยุดกลางคันแล้วรันคำสั่งเดิม + `--resume` เพื่อทำต่อ
  - เทรนบน scene split ทีละ fold: `python Task4/train.py --run-name cnsRS2_sc_f0 --model convnext_small --split scene --fold 0 --epochs 15 --schedule cosine --micro-batch 32 --aug robust --multi-scale --lr-backbone 1.5e-4` (ConvNeXt-S ต้อง micro-batch 32 ถึงจะพอดี GPU 8 GB)
  - TTA: `python Task4/train.py --mode tta --run-name cnsRS2_sc_f0` คำนวณห้ามุมมองตั้งต้นทั้ง validation และ test ถ้าใส่ `--tta-views id,hflip,...` จะคำนวณเฉพาะมุมมองที่ยังไม่มีแล้วต่อท้ายไฟล์เดิม
- `oof.py` รวมผล validation ของทั้ง 5 fold เป็น OOF เทียบชุด TTA และรวมหลายสูตร ทุกการเลือกใช้ OOF เท่านั้น เช่น `python Task4/oof.py --group "cnt=cnt_sc_f{fold}" --group "cnsRS=cnsRS2_sc_f{fold}" --views rot8+zoom --search --write-pred outputs/task4_exp/pred/task4_predictions.csv` ต่อท้ายชื่อกลุ่มด้วย `@0,1` ถ้ากลุ่มนั้นมีแค่บาง fold
- `final/` คำทำนายของไฟล์ส่งสุดท้าย (conv4p) ดูหัวข้อไฟล์ส่งสุดท้ายข้างบน
- `predict.py` ทายภาพในโฟลเดอร์ใดก็ได้ด้วย checkpoint ที่เทรนแล้ว และแพ็กโมเดลไว้ส่งต่อให้คนอื่น (ดูวิธีใช้ที่หัวไฟล์)
- `stress.py` ทำภาพ validation ให้เสียทีละแบบ 15 แบบ (JPEG, ความละเอียดต่ำ, เบลอ, noise, มืด ฯลฯ) แล้ววัด Macro F1 ใหม่ ตารางบอกด้วยว่าแบบไหนสูตรใดเคยเห็นตอนเทรน: `python Task4/stress.py --runs cnt_sc_f0,cnsRS2_sc_f0 --mix ens=cnt_sc_f0+cnsRS2_sc_f0`
- ตัวหลักเดิม (ResNet18) ยังรันได้ด้วย `python run.py train --tasks 4 --epochs 5 --batch-size 128` โค้ดอยู่ที่ `pis/image.py`
- เตรียมภาพ: `python prepare_images.py --download`
