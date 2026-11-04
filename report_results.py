"""Build a short measured-results report after every task has produced metrics."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from pis.common import DEFAULT_OUTPUT, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    out=args.output_dir
    m={i:json.loads((out/f'task{i}_metrics.json').read_text(encoding='utf-8')) for i in range(1,6)}
    manifest=json.loads((out/'submission_manifest.json').read_text(encoding='utf-8'))
    lines=['# ผลทดลอง Player Intelligence System',
           '', 'ผลต่อไปนี้วัดจากการรันในเครื่อง ยังไม่ได้ส่ง Kaggle และไม่ใช่คะแนน leaderboard', '',
           '| งาน | วิธีประเมิน | ผลจริง |', '|---|---|---|',
           f'| 1 | OOF {m[1]["folds"]}-fold, macro F2, threshold 0.5 | {m[1]["fixed_0_5_oof_score"]:.6f} |',
           f'| 1 | ปรับ threshold บน OOF เดิม มีความเอนเอียง | {m[1]["oof_score"]:.6f} |',
           f'| 2 | OOF {m[2]["folds"]}-fold, macro F1 | {m[2]["oof_score"]:.6f} |',
           f'| 3 | OOF {m[3]["folds"]}-fold, คะแนนตามสูตรการแข่งขัน | {m[3]["oof_score"]:.6f} |',
           f'| 4 | holdout ภาพ, macro F1 | {m[4]["validation_macro_f1"]:.6f} |',
           '| 5 | ไม่มีคำตอบกำกับสำหรับตรวจสอบ | คำนวณ macro F3 ไม่ได้ |', '',
           f'Task 1 ใช้แถวที่มี label {m[1]["n_labeled"]:,} แถว แยกแถวไม่มี label {m[1]["n_unlabeled_dropped"]:,} แถวออก เกณฑ์ที่เลือกเท่ากับ {m[1]["threshold"]:.6f} คะแนนที่ปรับ threshold มีความเอนเอียงจากการเลือกบน OOF ชุดเดียวกับที่รายงาน', '',
           f'Task 3 มี MAE {m[3]["mae"]:,.2f} บาทต่อผู้เล่น เทียบกับ median baseline {m[3]["baseline_mae"]:,.2f} บาท คะแนนใช้ 1/(1 + sum_absolute_error/sum_target) จึงยิ่งสูงยิ่งดี', '',
           f'Task 4 ใช้ภาพฝึก {m[4]["train_size"]:,} ภาพ และตรวจสอบ {m[4]["validation_size"]:,} ภาพ เก็บโมเดล epoch {m[4]["best_epoch"]} ภาพที่ byte เหมือนกันอยู่ฝั่งเดียวกัน แต่ยังไม่รับประกันว่าภาพใกล้ซ้ำหรือภาพจากวิดีโอเดียวกันถูกแยกกลุ่มครบ คะแนนใช้ holdout เดียวเลือก epoch', '',
           f'Task 5 ตั้งสมมติฐานแจ้งเตือน {m[5]["contamination_assumption"]:.0%} หรือ {m[5]["flagged_rows"]:,} บัญชี ตัวเลขนี้เป็นเกณฑ์ที่ตั้งไว้ ไม่ใช่สัดส่วนผิดปกติที่มีเฉลยยืนยัน', '',
           '![Measured validation results](validation_results.png)', '',
           '## ไฟล์ส่ง', '',
           f'ตรวจในเครื่องผ่าน {manifest["rows"]:,} แถว คอลัมน์ครบ ไม่มีค่าว่าง ยอดใช้จ่ายเป็นจำนวนไม่ติดลบ และ Task 4 ใช้ชื่อเกมจาก CSV จริง', '',
           f'รูปแบบตัวคั่นตามแม่แบบ: `{manifest["separator"]}`; SHA256: `{manifest["sha256"]}`', '',
           'การจับคู่ใช้แถวที่ i ของ test.csv แต่ละงานกับ ANS แถวที่ i เพราะไม่มีตารางรหัสเชื่อมให้ ควรยืนยันลำดับ ANS และรูปแบบ label Task 4 กับตัวตรวจการแข่งขัน การผ่านในเครื่องยังไม่ยืนยันว่า Kaggle ยอมรับไฟล์', '',
           'โจทย์ Task 5 ระบุให้เรียนรู้แบบไม่มีผู้สอนจากข้อมูลที่ให้ แต่สไลด์กติกาทั่วไปห้ามฝึกบน test โค้ดตีความว่า Task 5 เป็นข้อยกเว้นเฉพาะที่ใช้ข้อมูลไม่มี label ต้องยืนยันกับ TA ก่อนส่ง', '',
           'ยังคำนวณคะแนนรวมที่เทียบกับ baseline 0.68 หรือ challenge 0.75 ไม่ได้ เนื่องจากไม่มีเฉลย Task 5 และยังไม่ยืนยันน้ำหนักแต่ละงาน', '',
           '## ทดลองต่อจากหลักฐาน', '',
           '- ตรวจ false positives/false negatives ของ Task 1 โดยดู threshold 0.5 ควบคู่เกณฑ์ที่เลือก และใช้ holdout อิสระหากต้องการยืนยันประโยชน์ของการปรับเกณฑ์',
           '- ดูคู่คลาสที่สับสนของ Task 2 ก่อนทดลอง class weights บน split เดิม',
           '- แยก MAE ของผู้ไม่ใช้จ่ายและผู้ใช้จ่ายสูงใน Task 3 ก่อนเปรียบเทียบโมเดลสองขั้น',
           '- ตรวจภาพใกล้ซ้ำและตัวอย่างผิดของ Task 4 ก่อนเพิ่ม epoch',
           '- เปรียบเทียบจำนวนแจ้งเตือน Task 5 เมื่อเปลี่ยน contamination โดยไม่อ้างว่าคะแนนดีขึ้นหากยังไม่มี label', '',
           'ไฟล์ JSON ที่อยู่ข้างรายงานนี้เก็บผลราย fold, confusion matrix, feature importance และค่าตั้งของการทดลอง ใช้ประกอบรายงานวิชา โดยเพิ่มบทบาทสมาชิกและการอภิปรายผลของทีมเอง']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(['T1 F2 fixed','T1 F2 tuned*','T2 F1 OOF','T4 F1 holdout'],
                [m[1]['fixed_0_5_oof_score'],m[1]['oof_score'],m[2]['oof_score'],m[4]['validation_macro_f1']],color='#236c9c')
    axes[0].set(title='Classification validation scores',ylabel='Macro F score (0 to 1)',xlabel='Task / validation method',ylim=(0,1))
    axes[0].tick_params(axis='x',labelrotation=20)
    axes[1].bar(['Median baseline',m[3]['engine']+' OOF'],[m[3]['baseline_mae'],m[3]['mae']],color='#236c9c')
    axes[1].set(title='Task 3 spending absolute error',ylabel='MAE (THB per player)',xlabel='Model')
    history=m[4]['history']
    axes[2].plot([h['epoch'] for h in history],[h['validation_macro_f1'] for h in history],marker='o',label='Validation macro F1')
    axes[2].set(title='Task 4 epoch selection',xlabel='Training epoch (count)',ylabel='Macro F1',ylim=(0,1))
    axes[2].legend()
    fig.suptitle('Source: local validation metrics; *T1 tuned score is optimistically selected on OOF')
    fig.tight_layout();fig.savefig(out/'validation_results.png',dpi=160);plt.close(fig)
    (out/'RESULTS_TH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'results_summary.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),
        'task1_macro_f2_fixed':m[1]['fixed_0_5_oof_score'],'task1_macro_f2_tuned_optimistic':m[1]['oof_score'],
        'task2_macro_f1':m[2]['oof_score'],'task3_score':m[3]['oof_score'],'task3_mae_thb':m[3]['mae'],
        'task4_holdout_macro_f1':m[4]['validation_macro_f1'],'task5_macro_f3':None,
        'submission_rows':manifest['rows'],'uploaded_to_kaggle':False})
    print(out/'RESULTS_TH.md')


if __name__=='__main__':main()
