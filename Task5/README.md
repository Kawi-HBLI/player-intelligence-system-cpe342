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

## รุ่น v2 (branch `task5-boost`): สิ่งที่พบและสิ่งที่เปลี่ยน

Task 5 มีแค่ `test.csv` ทุกตัวเลขในหัวข้อนี้มาจากไฟล์นั้นโดยตรง ไม่มี label และไม่ใช้ข้อมูลภายนอก ไม่ใช้ลำดับแถวหรือเลข `player_id`

### anomaly ที่พบใน EDA

แถวปกติมีลักษณะชัดมาก ฟีเจอร์แต่ละตัวแทบไม่สัมพันธ์กันเลย (Spearman ระหว่างค่าเฉลี่ย 4 ช่วงของทุกคู่ฟีเจอร์ไม่เกิน 0.03) และค่าช่วงที่ 1 ถึง 4 ของคนเดียวกันสัมพันธ์กันสูง (0.8 ถึง 0.99) ส่วน anomaly แยกเป็นกลุ่มที่ไม่ทับกันเลย 8 กลุ่ม รวม 2,049 แถว หรือ 7.9% ของไฟล์

| กลุ่ม | จำนวนแถว | ลักษณะ | แยกจากแถวปกติได้แค่ไหน |
|---|---|---|---|
| A ทักษะพุ่งช่วงท้าย | 310 | ช่วงที่ 4 `skill_rating`, `skill_trajectory`, `kda_mean`, `accuracy_mean`, `friend_adds`, `friend_removes` พุ่ง `reaction_time_mean` ลด `device_count_4` และ `guild_changes_4` สูงขึ้น | มีช่องว่างชัด (robust z ของ `skill_rating_4` ไม่มีค่าระหว่าง 5 ถึง 8 เลย) |
| B ย้ายพิกัดแล้วโอนเงิน | 306 | พิกัดกระโดดในช่วง 3 หรือ 4, `location_changes_4` และ `currency_transfers_4` พุ่ง `play_pattern_consistency_4` ลด, `password_changed` = 1 ทุกแถว, `mass_item_sale` = 1 ราว 70% | ชัด ยกเว้นแถวที่ช่องสำคัญเป็นค่าว่าง |
| C ย้ายพิกัดแล้วซื้อของ | 194 (+17) | พิกัดกระโดด, `purchase_frequency_4` และ `purchase_amount_mean_4` พุ่ง, `vpn_usage` = 1 ทุกแถว (อีก 17 แถวมีแค่พิกัดกระโดดเพราะช่องซื้อของว่าง) | ชัด |
| D ค่ากระจายต่ำผิดปกติ | 207 | `ip_hash_entropy`, `kda_std`, `login_time_variance`, `session_duration_std` ต่ำทั้ง 4 ช่วงพร้อมกัน, `device_count_4` = 1 | เป็นเกาะ 1 มิติที่มีช่องว่างทุกตัว |
| E กิจกรรมต่ำ | 265 | `activity_level` เฉลี่ย 0.14 ถึง 0.27 (แถวปกติต่ำสุด 0.40) | เกาะ 1 มิติ |
| F ไม่คุยแต่ย้ายกิลด์บ่อย | 263 | `communication_frequency` ต่ำมาก และ `guild_changes` สูงกว่าแถวปกติ (ช่วงที่ 1 และ 4 เป็น 2 หรือ 3 เสมอ) | เกาะ 1 มิติ |
| G พฤติกรรมไม่คงที่ | 246 | `behavior_stability` 0.20 ถึง 0.40 (แถวปกติ 0.70 ขึ้นไป) | เกาะ 1 มิติ แต่ตรวจไม่ได้ถ้าช่องนี้ว่าง (ราว 13 แถว) |
| H กลุ่มหลายตัวแปร | 241 (EM ประเมิน 244) | `accuracy_std`, `kda_std`, `performance_stability` สูงพร้อมกัน แต่ละตัวยังอยู่ในช่วงปกติ | ดูทีละตัวไม่เห็น ต้องดูคู่ (มุมบนขวาของ `accuracy_std` กับ `kda_std` มีแถวมากกว่าที่ความเป็นอิสระทำนาย 3.8 เท่า z = 21.8) |

หลังตัด 8 กลุ่มนี้ออก แถวที่เหลือ 23,840 แถวไม่มีโครงสร้างให้ตรวจเจอเพิ่ม ลองไล่ทุกคู่ของฟีเจอร์ระดับ ฟีเจอร์การเปลี่ยนแปลงช่วง และ security flags (ราว 7,000 คู่) ค่า z สูงสุดเท่ากับที่สุ่มล้วนให้ได้ และพลังงาน shift ของแถวที่เหลือก็ตรงกับ null ที่สลับคอลัมน์

### detector เดิม (v1) ทำได้แค่ไหน

v1 profile strong (คอลัมน์ task5 ในไฟล์ทีมตอนนี้) ปักธง 2,140 แถว ครอบคลุม 2,000 จาก 2,049 แถวของ 8 กลุ่ม พลาดกลุ่ม H 39 แถว และกลุ่ม B/C ที่มีช่องว่าง 10 แถว อีก 140 ธงอยู่นอกทุกกลุ่ม เกือบทั้งหมดมาจาก residual HGB density ratio และดูเป็นแถวสุ่ม (อัตราส่วนความหนาแน่นในมุมของกลุ่ม H ใกล้ 1) ผลยังแกว่งตาม seed มาก seed 342, 7, 2026 ได้ 2,140 / 2,082 / 1,982 ธง

### สิ่งที่เปลี่ยนใน v2 (`pis/anomaly.py`)

1. **shift view ใหม่ (`surprise_shift_score`)** แต่ละ family วัด |robust z| ของการเปลี่ยนช่วงที่ 4 และครึ่งหลัง แล้วแปลงเป็น surprise = -log ของสัดส่วนแถว background ที่มีค่าสูงกว่า ทำให้ family ที่หางหนาโดยธรรมชาติ (เช่น `skill_trajectory`) ได้คะแนนน้อยลงเองที่ z เท่ากัน คะแนนแถวคือผลรวม 4 family ที่ surprise สูงสุด รวม security flag ที่ยกขึ้นด้วย (-log ของอัตราการยก) แล้วปรับเทียบกับ null ที่สลับแต่ละ family อิสระกัน วน 3 รอบเพื่อเอากลุ่มที่ตรวจเจอออกจาก background และเฉลี่ย posterior จาก null 5 ชุด ละติจูดกับลองจิจูดมีสถิติของตัวเอง (ยังเห็นการกระโดดได้แม้อีกพิกัดว่าง) แต่นับรวมกับ haversine เป็น family `geo` เดียว เพราะสามตัวนี้ขยับพร้อมกันแม้ในแถวปกติ ถ้านับแยกจะได้ธงเกินราว 25 ถึง 95 แถว
2. **dependency view ใหม่ (`dependency_group_score`)** แทน residual HGB density ratio หาคู่ฟีเจอร์ที่มุมหาง 5% หรือ 10% มีแถวมากเกินความเป็นอิสระ (z ≥ 6) จับคู่เป็นกลุ่มด้วย maximal clique ของโหนด (ฟีเจอร์, ทิศหาง) ใช้ kNN density ratio เทียบ null สลับคอลัมน์หาสมาชิกชั่วคราว เติมฟีเจอร์อื่นที่สมาชิกเลื่อนชัด (ในข้อมูลจริงได้ `performance_stability` เพิ่ม z = 15.6) แล้วให้ posterior สุดท้ายด้วย EM แบบ background อิสระ + Gaussian หนึ่งก้อนใน probit-copula space ได้ posterior ที่ลดลงตามระยะห่างจากกลุ่ม แทนที่ราบที่ isotonic calibration ชอบสร้าง
3. island view และการเลือก k ด้วย expected Macro F3 ใช้ของเดิม
4. ข้อมูลน้อยกว่า 10,000 แถว shift view ถอยไปใช้ hinge energy ของ v1 เพราะ isotonic calibration ที่หางยังแกว่ง และ dependency view ต้องมี background อย่างน้อย 2,000 แถว

### ผลตรวจโดยไม่มี label

- ข้อมูลจริง: v2 ปักธง 2,097 แถว ครอบคลุม 2,046 จาก 2,049 แถวของ 8 กลุ่ม มีธงนอกกลุ่ม 51 แถว (47 แถวเป็นขอบของกลุ่ม H ที่ EM ให้ posterior 0.16 ถึง 0.5) ผลเหมือนกันทุกแถวใน seed 342, 7, 2026 ทั้ง profile quick, balanced, strong (ต่างกันมากสุด 4 แถว) ประเมิน prevalence ได้ 7.92%
- ถ้าสมมติว่า 8 กลุ่มนี้คือ label จริง Macro F3 ของ v1 = 0.983 และ v2 = 0.997
- semi-synthetic (`Task5/synthetic_eval_v2.py`): ใช้แถวที่ v2 ให้ posterior ต่ำกว่า 0.01 (ราว 23,800 แถว) เป็นแถวปกติ แล้วฉีด anomaly 4 แบบ (shift หลาย family, เกาะ 1 มิติ, กลุ่มมุมหลายตัวแปร, shift อ่อน 2 family) โดยสุ่ม family ทิศ และขนาดใหม่ทุกชุด 3 seed × prevalence 5/8/10% แต่ละ detector เลือก k เอง ผลเฉลี่ย Macro F3 v1 = 0.786, v2 = 0.884, v2 รวมกับ v1 = 0.878 และ v2 ชนะ v1 ทั้ง 9 ชุด false positive ของสองรุ่นพอกัน (245 กับ 234 แถวโดยเฉลี่ย) แต่ v2 จับได้มากกว่ามาก recall ของ shift หลาย family 0.998 กับ 0.956 กลุ่มมุม 0.951 กับ 0.530 shift อ่อน 0.38 กับ 0.02 ส่วนเกาะ 1 มิติใช้ view เดียวกันจึงได้เท่ากัน (0.77) การรวม v1 เข้าไปเพิ่มแต่ false positive จึงแพ้ v2 ล้วน 8 ใน 9 ชุด

### ใช้งาน

```bash
python Task5/train.py --data-root ../cpe342-kaggle-main --profile strong --contamination auto
python Task5/train.py --data-root ../cpe342-kaggle-main --profile strong --t5-method v1
python Task5/synthetic_eval_v2.py ../cpe342-kaggle-main
```

v2 เป็นค่าเริ่มต้น ใช้เวลาราว 10 วินาที (v1 ราว 1 นาที) `task5_metrics.json` มีรายการกลุ่มหลายตัวแปรที่เจอใน `view_summary.dependency_groups`

### ข้อจำกัด

- ตัวเลขทั้งหมดอิงสมมติฐานว่า label จริงคือกลุ่มที่ข้อมูลแยกให้เห็น ถ้าผู้จัดติดป้าย anomaly ให้แถวที่ไม่มีร่องรอยในข้อมูลเลย ไม่มี detector ไหนจับได้ และการปักธงเพิ่มแบบสุ่มก็ไม่ช่วย เพราะอัตรา anomaly ในแถวที่เหลือต้องเกินราว 16% การปักธงเพิ่มถึงจะคุ้มกับ Macro F3
- dependency view หาเฉพาะกลุ่มที่อยู่ในมุมหางของคู่ฟีเจอร์ กลุ่มที่อยู่กลางการกระจายจะไม่เห็น
- แถวในกลุ่ม G ที่ `behavior_stability` ว่างตรวจไม่ได้
