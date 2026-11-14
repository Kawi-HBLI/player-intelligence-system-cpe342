# Task 5: Account Security Monitoring

ตรวจหาบัญชีที่มีพฤติกรรมผิดปกติ (1) แบบ unsupervised และปรับจุดตัดเชิงความน่าจะเป็นให้เหมาะกับเกณฑ์วัด **Macro F3** (ให้ recall มีน้ำหนักมากกว่า precision 9 เท่าในแต่ละคลาส)

## ข้อมูลและลักษณะความผิดปกติที่พบในเชิงโครงสร้าง (Unsupervised)

- มีเฉพาะ `test.csv` 25,889 แถว ไม่มี train และไม่มี label (`player_id` + 122 ฟีเจอร์)
- ฟีเจอร์ส่วนใหญ่เป็นอนุกรมเวลา 4 ช่วง (`_1` ถึง `_4`) พร้อมค่าว่างบางส่วน และ security flags 4 ตัว
- ในข้อมูลไร้ป้ายกำกับมีความผิดปกติหลายรูปแบบปะปนกัน ทั้งการเปลี่ยนพฤติกรรมฉับพลันช่วงท้าย/ครึ่งหลัง, การกระโดดของพิกัดล็อกอินข้ามช่วงเวลา, กลุ่มย่อยที่แยกตัวเป็น island ในการแจกแจง 1 มิติ, และกลุ่มความสัมพันธ์ร่วมหลายตัวแปรที่ค่าเฉลี่ยเดี่ยวไม่หลุดกรอบ

## สถาปัตยกรรมโมเดล (`pis/anomaly.py`)

ใช้ระบบให้คะแนน **3 มุมมองแบบ Empirical Bayes density-ratio** ที่เป็น unsupervised ล้วน (ไม่ใช้ label หรือ rule เฉพาะคอลัมน์):

1. **Topological Island Score (`topological_island_score`)**: ตรวจจับกลุ่มความหนาแน่นย่อยที่แยกตัวออกจากกลุ่มหลักด้วยร่องว่างความหนาแน่น (saddle valley) บนฟีเจอร์ต่อเนื่อง ข้ามฟีเจอร์ไม่ต่อเนื่อง (discrete) เพื่อไม่เกิดผลบวกลวง
2. **Calibrated Temporal Shift Score (`calibrated_shift_score`)**:
   - ตรวจจับความสัมพันธ์เชิงสเกลอัตโนมัติ (multiplicative vs additive) ด้วย Spearman correlation แล้วแปลง `log1p` ก่อนวัดการเปลี่ยนช่วงเวลา `d4` และ `dh` เทียบสเกล `MAD` ที่ทนทานต่อสัดส่วนความผิดปกติเกิน 1%
   - คำนวณระยะกระโดดเชิงภูมิศาสตร์ `geo_hav` (log-haversine สูงสุดข้ามช่วงเวลา) แยกจากค่าพิกัดเฉลี่ยเพื่อไม่ให้ภูมิภาคการเล่นถูกจัดเป็น anomaly
   - ทำ **Two-Pass Empirical Bayes calibration** กับ column-permuted null เพื่อลดการปนเปื้อนของกลุ่ม shift รุนแรง
3. **Residual Bagged Density Ratio (`residual_density_ratio`)**:
   - แยกกลุ่มพื้นหลังสะอาดจาก Stage 1 แล้วเทรน 5-fold OOF `HistGradientBoostingClassifier` เทียบกับ out-of-fold column-permuted null ที่จับคู่เวกเตอร์ตรงกันทุก bag พร้อม **sparse 2–3-feature linear projections** เพื่อแก้ปัญหา depth-0 tree split symmetry เมื่อ marginal เหมือนกัน
4. **Posterior Fusion & Expected Macro-F3 Thresholding (`select_expected_macro_f3`)**:
   - รวมความน่าจะเป็นอิสระด้วย Noisy-OR และเลือกจำนวนแถว `k*` (เมื่อ `--contamination auto`) ที่ให้ค่าคาดหวัง **Macro F3** สูงสุดโดยอัตโนมัติ (หรือรับ `--contamination <rate>` แบบระบุสัดส่วนตรงได้)

## การใช้งาน

- เทรนและสร้างผลลัพธ์ (ค่าเริ่มต้น `--profile balanced --contamination auto`):
  ```bash
  python Task5/train.py --data-root ../cpe342-kaggle-main --profile strong --contamination auto
  ```
- ผลลัพธ์ล่าสุดใน [outputs/main](file:///c:/Users/User/Downloads/mLProject/outputs/main) (`--profile strong`): คาดการณ์สัดส่วนผิดปกติ `estimated_prevalence = 7.93%`, ปักธง `2,008` แถว (`7.76%`), คงไฟล์ Tasks 1–4 จาก `baseline/` ไว้ครบและผ่าน `run.py validate` แล้ว
- สคริปต์ [Task5/synthetic_eval.py](file:///c:/Users/User/Downloads/mLProject/Task5/synthetic_eval.py) และ unit tests ใน [tests/test_integrity.py](file:///c:/Users/User/Downloads/mLProject/tests/test_integrity.py) ใช้ตรวจสอบการทำงานของ detector บนข้อมูลจำลองที่ไม่มีความสัมพันธ์แฝง
