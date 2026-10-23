# Smart Traffic Monitoring — รายงานสัปดาห์ที่ 1: Dataset + Model

> ส่วนที่รับผิดชอบ: Dataset, Data Preparation, Image Processing, Train/Fine-tune Model, ทดสอบ Model
> Deliverable: **Dataset + Initial Model + Experimental Result**
> เวอร์ชันสำหรับ GitHub: วางไว้ที่ root ของ repo ระดับเดียวกับ `results/` และ `runs/`

---

## สรุปผลสั้น ๆ (สำหรับสไลด์ / บทคัดย่อ)

- ใช้ dataset **Vehicle Counting v5** จาก Roboflow ซึ่งเป็นภาพกล้องจราจร มี **2,613 ภาพ 27,649 กล่อง 4 class** (bus, car, motorcycle, truck)
- Fine-tune **YOLO11n** 50 epoch บน Colab T4 ใช้เวลา 31 นาที
- ผลบน **Test set** (260 ภาพที่โมเดลไม่เคยเห็น): **mAP50 = 0.905, mAP50-95 = 0.676, Precision = 0.858, Recall = 0.836**
- โมเดลเร็วพอสำหรับวิดีโอ: inference **6.8 ms/ภาพ** บน T4 และทั้ง pipeline วิดีโอรันได้ **20.9 FPS**
- **Median filter** กู้ความแม่นยำจากภาพที่มี noise กลับมาได้ **94%** (mAP50 จาก 0.611 กลับเป็น 0.888)
- **Gamma + CLAHE ทำให้ภาพกลางคืนแย่ลง** (mAP50 จาก 0.791 ลดเป็น 0.623) จึงตัดสินใจไม่ใช้ใน pipeline
- เทียบกับ dataset ชุดแรก (12 class ซ้ำซ้อน, val mAP50 0.440) การเปลี่ยน dataset ทำให้ val mAP50 ขึ้นเป็น **0.904**

---

## 1. Problem & Objective (ร่างให้กลุ่มปรับ)

**Problem:** การนับและจำแนกประเภทรถบนถนนยังพึ่งคนนับหรือเซนเซอร์ฝังถนนซึ่งมีต้นทุนสูง ขณะที่กล้อง CCTV ตามทางแยกมีอยู่แล้วจำนวนมากแต่ยังไม่ได้ถูกนำข้อมูลมาวิเคราะห์อัตโนมัติ ถนนในเอเชียตะวันออกเฉียงใต้ยังมีมอเตอร์ไซค์หนาแน่น ซึ่งเป็นวัตถุขนาดเล็กและบังกันเอง ทำให้ตรวจจับยากกว่ารถยนต์

**Objective (ส่วน Dataset + Model):**
1. เตรียม dataset ภาพกล้องจราจรที่มี 4 ประเภทรถ รวมมอเตอร์ไซค์ และแบ่ง Train/Validation/Test อย่างเป็นระบบ
2. ทดลองเทคนิค Image Processing และ**วัดผลเชิงตัวเลข**ว่าเทคนิคใดช่วยการตรวจจับได้จริง
3. Fine-tune YOLO11 ให้ได้ mAP50 ≥ 0.85 บน test set และเร็วพอสำหรับวิดีโอ
4. ส่งมอบ `best.pt` + `preprocessing.py` ให้ทีมสัปดาห์ 2 ใช้ทำระบบ Tracking + นับรถ

---

## 2. Dataset

### 2.1 แหล่งข้อมูล

| รายการ | รายละเอียด |
|---|---|
| ชื่อ | Vehicle Counting (เวอร์ชัน 5, "ATCS-V5") |
| ที่มา | Roboflow Universe — https://universe.roboflow.com/pram/vehicle-counting-flusm |
| License | CC BY 4.0 (ต้องใส่เครดิตใน README) |
| ลักษณะภาพ | ภาพจากกล้อง CCTV ทางแยก มีทั้งกลางวันและกลางคืน (จากชื่อเวอร์ชันและภาพตัวอย่าง น่าจะเป็นกล้อง ATCS ในยอกยาการ์ตา อินโดนีเซีย) |
| ขนาดภาพ | 640×640 ทุกภาพ (Roboflow resize แบบ stretch ไว้แล้ว) |
| Format | YOLOv11 (`class x_center y_center width height`, normalize 0–1) |

**เหตุผลที่เลือก:** มี class ตรงกับงานนับรถ 4 ประเภท **มีมอเตอร์ไซค์** ซึ่งสำคัญกับถนนแบบไทย และเป็นมุมกล้อง CCTV แบบเดียวกับที่ระบบจะใช้งานจริง ภาพไม่ถูก augment มาก่อน ค่าที่วัดได้จึงไม่ถูกภาพซ้ำทำให้สูงเกินจริง

### 2.2 จำนวนภาพและการแบ่ง Train / Validation / Test

| Split | จำนวนภาพ | % | จำนวนวัตถุ | วัตถุ/ภาพ |
|---|---:|---:|---:|---:|
| Train | 1,830 | 70% | 19,445 | 10.6 |
| Validation | 523 | 20% | 5,414 | 10.4 |
| Test | 260 | 10% | 2,790 | 10.7 |
| **รวม** | **2,613** | 100% | **27,649** | 10.6 |

- ไม่มีภาพที่ไม่มีวัตถุ (background image = 0)
- **Test set ใช้วัดผลครั้งสุดท้ายเท่านั้น** ไม่ได้ใช้เลือก epoch หรือปรับค่าใด ๆ

### 2.3 จำนวนวัตถุแต่ละ Class

| Class | Train | Val | Test | รวม | สัดส่วน |
|---|---:|---:|---:|---:|---:|
| motorcycle | 8,985 | 2,414 | 1,264 | 12,663 | 45.8% |
| car | 6,938 | 2,017 | 1,006 | 9,961 | 36.0% |
| truck | 2,432 | 677 | 373 | 3,482 | 12.6% |
| bus | 1,090 | 306 | 147 | 1,543 | 5.6% |


![รูปที่ 1 จำนวนวัตถุแต่ละ class, ขนาดวัตถุ และการกระจายขนาด bounding box](results/dataset_statistics.png)
*รูปที่ 1 จำนวนวัตถุแต่ละ class, ขนาดวัตถุ และการกระจายขนาด bounding box*


**วิเคราะห์:**
- **Class imbalance 8.2 เท่า** (motorcycle 12,663 กล่อง กับ bus 1,543 กล่อง) ซึ่งสะท้อนสภาพถนนจริง แต่ทำให้โมเดลเห็นตัวอย่างรถบัสน้อย
- **ขนาดวัตถุ** (นิยามแบบ COCO): medium 52.5%, **small (<32×32 px) 32.2%**, large 15.3% วัตถุราว 1 ใน 3 เป็นวัตถุเล็ก ส่วนใหญ่เป็นมอเตอร์ไซค์และรถที่อยู่ไกลกล้อง
- **ความสว่าง:** ภาพมืด (ค่า V เฉลี่ยใน HSV < 80) มีเพียง **1.4%** ของ train ข้อมูลกลางคืนจึงน้อยมาก

![รูปที่ 2 การกระจายความสว่างและ contrast ของภาพ](results/dataset_brightness.png)
*รูปที่ 2 การกระจายความสว่างและ contrast ของภาพใน train set*


### 2.4 ตัวอย่างข้อมูล


![รูปที่ 3 ตัวอย่างภาพใน dataset พร้อมกล่อง ground truth แยกสีตาม class](results/dataset_samples.jpg)
*รูปที่ 3 ตัวอย่างภาพใน dataset พร้อมกล่อง ground truth แยกสีตาม class*


### 2.5 Data Preparation ที่ทำ

1. โหลดผ่าน Roboflow API ในรูปแบบ YOLOv11
2. **แก้ path ใน `data.yaml`** เดิมเป็น `../train/images` ซึ่ง Ultralytics หาไฟล์ไม่เจอ จึงเขียนใหม่เป็น `data_fixed.yaml` แบบ absolute path
3. ตรวจว่ามี test split ครบ (notebook มีโค้ดแบ่ง 10% จาก train ให้อัตโนมัติถ้าไม่มี แต่ชุดนี้มีอยู่แล้ว)
4. ตรวจ label ทุกไฟล์ รองรับทั้ง bbox และ polygon และนับสถิติจำนวน ขนาด และความสว่าง

---

## 3. Image Processing

### 3.1 แนวคิดการเลือกเทคนิค

ไม่ได้ใส่ทุกเทคนิคที่มี แต่เลือกจาก**ปัญหาจริงของกล้อง CCTV** แล้ว**วัดผลด้วย mAP** ว่าเทคนิคนั้นช่วยการตรวจจับได้จริงหรือไม่ ก่อนตัดสินใจนำไปใช้ใน pipeline

| สถานะ | เทคนิค | ฟังก์ชันใน `src/preprocessing.py` | ปัญหาที่แก้ | ผล |
|---|---|---|---|---|
| ✅ ใช้จริง | **Median Filter 3×3** | `median_filter()` | noise แบบจุด (salt & pepper) จากสัญญาณกล้องเสีย / ส่งภาพผ่านเครือข่าย | mAP50 **+0.277** |
| ✅ ใช้จริง | **Resize แบบ letterbox 640×640** | `letterbox()` | YOLO ต้องการ input 640×640 ถ้า resize ตรง ๆ สัดส่วนรถจะบิด จึงย่อตามสัดส่วนแล้วเติมขอบสีเทา | ขั้นตอนบังคับก่อนเข้าโมเดล |
| ❌ ทดลองแล้วไม่ใช้ | **Gamma Correction + CLAHE** | `auto_gamma()`, `clahe()` | ภาพมืดตอนกลางคืน / contrast ต่ำ | mAP50 **−0.168** |

**หลักการทำงานโดยย่อ**
- **Median Filter:** แทนค่าพิกเซลด้วย**ค่ากลาง (median)** ของพิกเซลรอบ ๆ ในหน้าต่าง 3×3 พิกเซล noise ที่ขาวจัดหรือดำจัดจะถูกแทนที่ทั้งหมด และต่างจาก Gaussian blur ตรงที่**ไม่ทำให้ขอบรถเบลอ** เพราะค่ากลางเป็นค่าที่มีอยู่จริงในบริเวณนั้น
- **Letterbox:** ย่อภาพด้วยอัตราส่วน `r = 640 / max(H, W)` ให้ด้านยาวเท่ากับ 640 แล้วเติมขอบสีเทา (114,114,114) ให้เป็นสี่เหลี่ยมจัตุรัส รถจึงไม่ถูกยืดหรือบีบ
- **Gamma Correction:** ปรับความสว่างแบบไม่เป็นเส้นตรง `out = 255 × (in/255)^(1/γ)` ถ้า γ > 1 ส่วนมืดจะสว่างขึ้นมากกว่าส่วนที่สว่างอยู่แล้ว ระบบเลือก γ อัตโนมัติให้ความสว่างเฉลี่ยใกล้ 45%
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** แบ่งภาพเป็นตาราง 8×8 แล้วทำ histogram equalization ทีละช่อง บนช่อง L ของ LAB (สีไม่เพี้ยน) และจำกัด contrast (clip limit 2.0) เพื่อไม่ให้ขยาย noise มากเกินไป

**ทำไมต้อง Letterbox:** เฟรมวิดีโอจริงไม่ใช่สี่เหลี่ยมจัตุรัส (เช่น คลิป `traffic_video_1.mp4` เป็นแนวตั้ง 1080×1920) ถ้า resize ตรง ๆ เป็น 640×640 รถจะถูกยืดออกด้านข้าง ทำให้รูปทรงต่างจากที่โมเดลเรียนมา ส่วน letterbox รักษาสัดส่วนเดิมแล้วเติมขอบสีเทาแทน (รูปที่ 4)

![รูปที่ 4 เฟรมจาก `traffic_video_1.mp4` (แนวตั้ง 1080×1920): ภาพต้นฉบับ, resize ตรง ๆ เป็น 640×640 (รถบิด) และ letterbox 640×640 (รักษาสัดส่วน)](results/letterbox_demo.png)
*รูปที่ 4 เฟรมจาก `traffic_video_1.mp4` (แนวตั้ง 1080×1920): ภาพต้นฉบับ, resize ตรง ๆ เป็น 640×640 (รถบิด) และ letterbox 640×640 (รักษาสัดส่วน)*

![รูปที่ 5 สถานการณ์จำลอง: noise แบบ salt & pepper → Median filter และภาพกลางคืน → Gamma + CLAHE](results/image_processing_scenarios.png)
*รูปที่ 5 สถานการณ์จำลอง: noise แบบ salt & pepper → Median filter และภาพกลางคืน → Gamma + CLAHE*

![รูปที่ 6 Histogram ความเข้มของภาพที่มืดที่สุดใน dataset ก่อนและหลัง Gamma + CLAHE](results/histogram_before_after.png)
*รูปที่ 6 Histogram ความเข้มของภาพที่มืดที่สุดใน dataset ก่อนและหลัง Gamma + CLAHE*

### 3.2 การทดลอง: Image Processing ช่วยการตรวจจับจริงไหม?

**วิธีทดลอง:** ใช้โมเดล `best.pt` ตัวเดียวกัน วัดบน test set 260 ภาพใน 5 สถานการณ์ โดยจำลองภาพเสียแล้วแก้ด้วย Image Processing
- **Night:** ลดความสว่าง (gamma 1.6 × 0.35) แล้วเติม Gaussian noise σ=6 (brightness เฉลี่ยจาก 113 เหลือ 31)
- **Noise:** เติม salt & pepper 4% ของพิกเซล

| สถานการณ์ | mAP50 | mAP50-95 | Precision | Recall | เทียบกับก่อนแก้ |
|---|---:|---:|---:|---:|---:|
| Original | **0.905** | 0.675 | 0.871 | 0.825 | — |
| Noise (salt & pepper) | 0.611 | 0.389 | 0.657 | 0.590 | — |
| Noise → **Median filter 3×3** | **0.888** | 0.658 | 0.868 | 0.809 | **+0.277** ✅ |
| Night | 0.791 | 0.546 | 0.812 | 0.696 | — |
| Night → Gamma + CLAHE | 0.623 | 0.414 | 0.674 | 0.575 | **−0.168** ❌ |

![รูปที่ 7 mAP50 บน test set ในแต่ละสถานการณ์ (ฟ้า = ภาพปกติ, แดง = ภาพเสีย, เขียว = หลังแก้ด้วย Image Processing)](results/robustness.png)
*รูปที่ 7 mAP50 บน test set ในแต่ละสถานการณ์ (ฟ้า = ภาพปกติ, แดง = ภาพเสีย, เขียว = หลังแก้ด้วย Image Processing)*

### 3.3 วิเคราะห์ผลและการตัดสินใจ

**1. Median filter: เลือกใช้** ✅
noise แบบจุดทำให้ mAP50 ตกจาก 0.905 เหลือ 0.611 พอผ่าน median filter 3×3 แล้วกลับขึ้นมาเป็น 0.888 กู้คืนได้ **94%** ของที่เสียไป และต่ำกว่าภาพปกติแค่ 0.017 เพราะ median filter ลบพิกเซลผิดปกติออกได้โดยแทบไม่ทำลายขอบรถ ซึ่งเป็นข้อมูลสำคัญที่ YOLO ใช้ตรวจจับ

**2. Gamma + CLAHE: ไม่ใช้** ❌ (ผลตรงข้ามกับที่คาดไว้)
mAP50 ของภาพกลางคืนลดจาก 0.791 เป็น 0.623 สาเหตุที่น่าจะเป็นคือ
- ภาพกลางคืนจริงและที่จำลองมี noise ปนอยู่ การยก gamma และ CLAHE **ขยาย noise ไปพร้อมกับรายละเอียด** ภาพจึงหยาบและ contrast จัดผิดธรรมชาติ
- ภาพหลังปรับสว่างเกินภาพต้นฉบับ (brightness 138 เทียบกับ 113) ภาพจึง**ต่างจากภาพที่โมเดลเคยเห็นตอน train** (distribution shift)
- โมเดลทนต่อความมืดได้ระดับหนึ่งอยู่แล้ว เพราะตอน train ใช้ HSV augmentation (`hsv_v=0.4`) ภาพมืดจึงตก mAP50 แค่ 0.114

**3. ข้อสรุปเชิงออกแบบระบบ**
- Pipeline ที่ใช้จริง: `ภาพจากกล้อง → (Median filter ถ้าภาพมี noise) → Letterbox 640 → YOLO11n`
- ปัญหากลางคืนควรแก้ที่**ข้อมูล** (เพิ่มภาพกลางคืนจริง / brightness augmentation ตอน train) ไม่ใช่ทำ preprocessing ตอน inference

> Image Processing ไม่ได้ช่วยเสมอไป ต้องเลือกให้ตรงกับชนิดของปัญหา และวัดผลจริงก่อนนำไปใช้

---

## 4. Computer Vision / AI Model

### 4.1 Algorithm: YOLO11 (Object Detection)

YOLO (You Only Look Once) เป็น one-stage detector ที่ทำนายกล่องและ class ของวัตถุทุกตัวในภาพจากการประมวลผลรอบเดียว จึงเร็วพอสำหรับวิดีโอ real-time

เลือก **YOLO11n** (nano, 2.58 ล้านพารามิเตอร์, 6.4 GFLOPs) เพราะ
- งานนับรถต้องประมวลผลวิดีโอต่อเนื่อง ความเร็วจึงสำคัญ
- เริ่มจาก weight ที่ pretrain บน COCO ซึ่งมี car, bus, truck, motorcycle อยู่แล้ว (**Transfer Learning**) จึงเรียนรู้ได้เร็วแม้มีข้อมูลแค่ 1,830 ภาพ

### 4.2 การตั้งค่าการ Train

| Parameter | ค่า |
|---|---|
| Base model | `yolo11n.pt` (COCO pretrained) |
| Epochs | 50 (early stopping patience 15 ไม่ได้ทำงาน เพราะ mAP ยังดีขึ้นจนถึง epoch สุดท้าย) |
| Image size | 640 |
| Batch | 16 |
| Optimizer | auto → AdamW, lr ≈ 0.00125, ลด lr แบบ linear |
| Augmentation | HSV (h 0.015, s 0.7, v 0.4), flip ซ้าย-ขวา 0.5, Mosaic 1.0 (ปิด 10 epoch สุดท้าย), scale 0.5, translate 0.1, RandAugment, random erasing 0.4 |
| Seed | 42 (deterministic) |
| Hardware | Google Colab, Tesla T4 |
| เวลา train | 0.51 ชั่วโมง (31 นาที) |

### 4.3 กราฟการ Train


![รูปที่ 8 Loss และ validation metrics ระหว่างการ train](results/training_curves.png)
*รูปที่ 8 Loss และ validation metrics ระหว่างการ train*

![รูปที่ 9 กราฟผลการ train ทั้งหมดจาก Ultralytics](results/results.png)
*รูปที่ 9 กราฟผลการ train ทั้งหมดจาก Ultralytics*


- loss ทั้ง box, class และ DFL ลดลงต่อเนื่องทั้ง train และ val **ไม่มีสัญญาณ overfitting** (val box loss ต่ำสุดที่ epoch สุดท้าย 0.986)
- val mAP50 ขึ้นเร็วในช่วงแรก (0.576 → 0.840 ที่ epoch 15) แล้วค่อย ๆ ขึ้นจนถึง **0.904** ที่ epoch 50
- **Best epoch = 50** (val mAP50-95 = 0.671) แปลว่าโมเดล**ยังไม่อิ่มตัว** ถ้า train นานขึ้น เช่น 100 epoch น่าจะดีขึ้นอีก

### 4.4 เปรียบเทียบกับการทดลองแรก (Dataset ชุดแรก)

| | Dataset ชุดแรก | Vehicle Counting v5 |
|---|---|---|
| จำนวน class | 12 (เช่น `big truck`, `truck-l-`, `truck-m-`, `small bus`, `bus-s-`) | 4 |
| ปัญหา | class ซ้ำกันเพราะใช้ชื่อ 2 แบบ, แยกรถตามขนาดซึ่งแยกด้วยตาได้ยาก, บาง class มีแค่ 8–49 กล่อง, ไม่มี motorcycle | — |
| หยุด train | early stop ที่ epoch 33 (best epoch 18) | ครบ 50 epoch |
| **val mAP50** | **0.440** | **0.904** |
| val mAP50-95 | 0.300 | 0.671 |

![รูปที่ 10 ผลการ train ของ dataset ชุดแรก (12 class)](runs/yolo11n_traffic/results.png)
*รูปที่ 10 กราฟการ train ของ dataset ชุดแรก (12 class): mAP50 ค้างอยู่ประมาณ 0.4 และหยุดที่ epoch 33*

**บทเรียน:** คุณภาพและการออกแบบ class ของ dataset มีผลต่อความแม่นยำมากกว่าการปรับโมเดล ใช้โมเดลและค่าตั้งเดียวกันทั้งสองรอบ แต่ mAP50 ต่างกันถึง 2 เท่า

---

## 5. Model Evaluation (Test set, 260 ภาพ / 2,790 วัตถุ)

### 5.1 ผลรวม

| mAP50 | mAP50-95 | Precision | Recall | F1 | Inference |
|---:|---:|---:|---:|---:|---:|
| **0.905** | **0.676** | 0.858 | 0.836 | 0.847 | 6.8 ms/ภาพ (T4) |

> ความเร็ว inference: **6.8 ms/ภาพ บน GPU T4** และประมาณ **239 ms/ภาพ บน CPU** ของ Colab (วัดจากการรันรอบที่ไม่มี GPU) ถ้า app สัปดาห์ 2–3 ต้องรันบน CPU ควรประมวลผลวิดีโอแบบข้ามเฟรม (เช่น `vid_stride`) หรือลดขนาดภาพ

- **mAP50 = 0.905** นับว่าตรวจถูกเมื่อกล่องทำนายทับกล่องจริง ≥ 50%
- **mAP50-95 = 0.676** เข้มงวดกว่า (เฉลี่ย IoU 0.50–0.95) วัดว่ากล่องวางตรงตำแหน่งแค่ไหน
- **Recall 0.836** หมายความว่ารถจริงทุก 100 คัน ตรวจเจอประมาณ 84 คัน ค่านี้สำคัญที่สุดสำหรับงาน "นับรถ"
- **Confidence threshold ที่ให้ F1 สูงสุด = 0.37** ส่งให้ทีมสัปดาห์ 2 ใช้เป็นค่าเริ่มต้น
- ค่า val (0.904) กับ test (0.905) ใกล้กันมาก แสดงว่าผลไม่ได้สูงเพราะบังเอิญตรงกับชุด validation


![รูปที่ 11 Precision-Recall curve บน test set](results/BoxPR_curve.png)
*รูปที่ 11 Precision-Recall curve บน test set*

![รูปที่ 12 F1-Confidence curve (F1 สูงสุดที่ conf = 0.37)](results/BoxF1_curve.png)
*รูปที่ 12 F1-Confidence curve (F1 สูงสุดที่ conf = 0.37)*


### 5.2 ผลแยกตาม Class

| Class | จำนวนใน test | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| bus | 147 | 0.895 | 0.815 | 0.918 | **0.751** |
| car | 1,006 | 0.882 | 0.865 | **0.921** | 0.691 |
| truck | 373 | 0.809 | 0.842 | 0.893 | 0.699 |
| motorcycle | 1,264 | 0.845 | 0.822 | 0.887 | **0.561** |

📊 `results/test_metrics_per_class.csv`

**วิเคราะห์:**
- **motorcycle มี mAP50-95 ต่ำที่สุด (0.561)** ทั้งที่มีข้อมูลมากที่สุด เพราะเป็นวัตถุเล็ก จอดชิดกันและบังกันเอง (ดูภาพ detection ที่มีมอเตอร์ไซค์อัดกันหน้าเส้นหยุด) โมเดล "เจอ" ได้ดี (mAP50 0.887) แต่วางกล่องให้พอดีเป๊ะได้ยาก ข้อผิดพลาดเพียงไม่กี่พิกเซลมีผลกับ IoU ของวัตถุเล็กมาก
- **bus มี mAP50-95 สูงสุด (0.751)** แม้ข้อมูลน้อยที่สุด เพราะเป็นวัตถุใหญ่ รูปทรงชัด วางกล่องได้ง่าย
- **truck มี Precision ต่ำสุด (0.809)** คือมีการทาย truck ผิดค่อนข้างมาก เพราะรูปร่างคล้ายรถบัสและรถกระบะ/รถตู้ (car)

### 5.3 Confusion Matrix และ Precision / Recall / AP ต่อ class

**วิธีอ่านตาราง** (แบบเดียวกับในชั้นเรียน): **แถว = ค่าจริง (Actual), คอลัมน์ = ที่โมเดลทำนาย (Predicted)** นับบน test set ที่ `conf = 0.37` ซึ่งเป็นค่าที่ใช้งานจริง

งาน object detection ต่างจาก classification ในสไลด์ตรงที่มีอีก 2 กรณี จึงเพิ่มช่องพิเศษ
- **คอลัมน์ "missed (FN)":** รถมีอยู่จริง แต่โมเดลไม่ตีกรอบเลย
- **แถว "no vehicle (FP)":** โมเดลตีกรอบ แต่ตรงนั้นไม่มีรถ (false alarm)

| ค่าจริง \ ทำนาย | bus | car | motorcycle | truck | missed (FN) | รวม (รถจริง) |
|---|---:|---:|---:|---:|---:|---:|
| **bus** | **118** | 4 | 0 | 2 | 23 | 147 |
| **car** | 1 | **865** | 3 | 22 | 115 | 1,006 |
| **motorcycle** | 0 | 4 | **1,045** | 0 | 215 | 1,264 |
| **truck** | 5 | 20 | 0 | **309** | 39 | 373 |
| **no vehicle (FP)** | 10 | 95 | 189 | 56 | — | |

![รูปที่ 13 Confusion matrix บน test set ที่ conf 0.37 (แถว = ค่าจริง, คอลัมน์ = ที่ทำนาย; ทแยงเขียว = ถูก, ตัวแดง = ผิด)](results/confusion_table.png)
*รูปที่ 13 Confusion matrix บน test set ที่ conf 0.37 (แถว = ค่าจริง, คอลัมน์ = ที่ทำนาย; ทแยงเขียว = ถูก, ตัวแดง = ผิด)*

**การคำนวณ** (ตัวอย่าง class car)
- **TP** = ช่องทแยง = **865**
- **FN** = ผลรวมแถว car − TP = 1 + 3 + 22 + 115 = **141** (รถยนต์จริงที่ทายเป็นคลาสอื่น หรือหาไม่เจอ)
- **FP** = ผลรวมคอลัมน์ car − TP = 4 + 4 + 20 + 95 = **123** (กรอบที่ทายว่า car แต่จริงเป็นคลาสอื่น หรือไม่มีรถ)

$$Precision_{car} = \frac{TP}{TP+FP} = \frac{865}{865+123} = 0.88 \qquad Recall_{car} = \frac{TP}{TP+FN} = \frac{865}{865+141} = 0.86$$

$$AP_{car} \approx Precision \times Recall = 0.88 \times 0.86 = 0.75$$

**ผลทุก class**

| Class | TP | FP | FN | Precision | Recall | AP ≈ P × R | AP50 จริง (PR curve) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bus | 118 | 16 | 29 | 0.881 | 0.803 | 0.707 | 0.918 |
| car | 865 | 123 | 141 | 0.876 | 0.860 | **0.753** | **0.921** |
| motorcycle | 1,045 | 192 | 219 | 0.845 | 0.827 | 0.698 | 0.887 |
| truck | 309 | 80 | 64 | **0.794** | 0.828 | 0.658 | 0.893 |
| **เฉลี่ย** | | | | 0.849 | 0.830 | 0.704 | 0.905 |

📊 `results/confusion_table.csv`, `results/precision_recall_ap_per_class.csv`

**วิเคราะห์**
1. **AP ≈ P × R ต่ำกว่า AP50 จริงทุก class** (0.70 กับ 0.905) เพราะสูตรในสไลด์ใช้ Precision และ Recall **ที่ threshold เดียว** (conf 0.37) ส่วน AP จริงคือ**พื้นที่ใต้กราฟ Precision–Recall ทุก threshold** (รูปที่ 11) สูตร P × R จึงใช้ประมาณค่าเพื่อทำความเข้าใจได้ แต่ค่าที่ใช้รายงานผลคือ AP50 / mAP50
2. **ความผิดพลาดส่วนใหญ่ไม่ใช่การสับสนระหว่างคลาส แต่เป็น "หาไม่เจอ" กับ "ทายเกิน"** ช่องสีแดงระหว่างคลาสมีน้อยมาก (รวม 61 จาก 2,790 คัน) ส่วนที่มากคือคอลัมน์ missed (392 คัน) และแถว no vehicle (350 กรอบ)
3. **motorcycle หาไม่เจอมากที่สุด (215 คัน = 17% ของมอเตอร์ไซค์ทั้งหมด)** เพราะเป็นวัตถุเล็กและจอดเบียดกัน และ**มี false alarm มากที่สุด (189 กรอบ)** ซึ่งส่วนหนึ่งเป็น**มอเตอร์ไซค์จริงที่ dataset ไม่ได้ label** (หัวข้อ 5.4 รูปที่ 16) ค่า Precision จริงจึงน่าจะสูงกว่าที่วัดได้
4. **truck มี Precision ต่ำสุด (0.794)** สับสนกับ car ไปกลับ (truck → car 20 คัน, car → truck 22 คัน) และกับ bus (5 คัน) เพราะรถกระบะ รถตู้ และรถบรรทุกเล็กหน้าตาคล้ายกัน
5. **bus เกือบไม่สับสนกับคลาสอื่นเลย** ผิดเป็นคลาสอื่นแค่ 6 คัน แต่หาไม่เจอ 23 คัน (16%) น่าจะเพราะมีข้อมูลรถบัสน้อยที่สุด (5.6% ของ dataset)

**เปรียบเทียบกับ confusion matrix ของ Ultralytics:** กราฟ `confusion_matrix.png` ที่ Ultralytics สร้างอัตโนมัติ (รูปที่ 14) นับที่ **conf 0.001** คือนับทุกกรอบแม้โมเดลมั่นใจแค่ 0.1% ช่อง background จึงมีตัวเลขสูงมาก (เช่น 11,048 กรอบที่ทายเป็น motorcycle) และสัดส่วนในแนวทแยงต่ำกว่าความจริง กราฟนี้ใช้ดูภาพรวมว่าคลาสไหนสับสนกับคลาสไหน แต่ไม่ควรเอามาคำนวณ Precision / Recall ตรง ๆ จึงใช้ตารางที่ conf 0.37 ด้านบนแทน

![รูปที่ 14 Confusion matrix (normalized) จาก Ultralytics ที่ conf 0.001 (แถว = Predicted, คอลัมน์ = True)](results/confusion_matrix_normalized.png)
*รูปที่ 14 Confusion matrix (normalized) จาก Ultralytics ที่ conf 0.001 (แถว = Predicted, คอลัมน์ = True)*

### 5.4 ทดสอบการตรวจจับและนับบนภาพ


![รูปที่ 15 ผลการตรวจจับบน test set 6 ภาพ (หัวภาพ: จำนวนที่ทำนาย / ground truth / จำนวนแยก class)](results/detection_grid.jpg)
*รูปที่ 15 ผลการตรวจจับบน test set 6 ภาพ (หัวภาพ: จำนวนที่ทำนาย / ground truth / จำนวนแยก class)*


| ภาพ | ทำนาย | Ground truth |
|---|---:|---:|
| 1 (กลางคืน) | 8 | 8 |
| 2 | 16 | 6 |
| 3 (ทางแยกกว้าง) | 17 | 24 |
| 4 | 11 | 12 |
| 5 (มอเตอร์ไซค์หนาแน่น) | 18 | 17 |
| 6 | 12 | 11 |
| **รวม** | **82** | **78** |

- ภาพกลางคืน (ภาพ 1) นับได้ตรง 8/8 แม้มีแสงไฟหน้ารถจ้า
- ภาพ 3 นับขาด 7 คัน เพราะรถที่อยู่ไกลกล้องเล็กมาก (ตรงกับปัญหา small object)
- ภาพ 2 ทำนาย 16 แต่ GT มีแค่ 6 เมื่อวางภาพ Ground truth เทียบกับ Prediction (รูปที่ 16) พบว่า **dataset label เฉพาะรถยนต์และรถบัส แต่ไม่ได้ label มอเตอร์ไซค์ในภาพนี้เลยสักคัน** ขณะที่ในภาพมีมอเตอร์ไซค์อย่างน้อย 7–8 คัน และโมเดลตรวจเจอได้เกือบครบ กรอบที่โมเดลทายผิดจริงมีเพียง 2 กรอบ คือ `car 0.47` และ `motorcycle 0.43` ที่ซ้อนอยู่บนตัวรถบัส
- ข้อค้นพบนี้อธิบายส่วนหนึ่งของ confusion matrix (หัวข้อ 5.3) ที่ false positive ในคอลัมน์ background ส่วนใหญ่เป็น motorcycle (47%) **บางส่วนเป็นมอเตอร์ไซค์จริงที่ dataset ไม่ได้ label** ค่า Precision ของ motorcycle จึงน่าจะต่ำกว่าความจริง

![รูปที่ 16 Ground truth vs Prediction](results/gt_vs_pred_example.jpg)
*รูปที่ 16 เทียบ Ground truth (ซ้าย, 6 กรอบ ไม่มีมอเตอร์ไซค์) กับ Prediction ของโมเดล (ขวา, conf 0.37)*


### 5.5 ทดสอบบนวิดีโอ: Tracking + นับรถข้ามเส้น (ต่อยอดสัปดาห์ 2)

- วิดีโอ `traffic_video/traffic_video_3.mp4` (1920×1080, 25 fps กล้องมุมสูงที่ทางแยกในเมือง) ประมวลผลครบทั้งคลิป 611 เฟรม (~24 วินาที)
- วิดีโอผลลัพธ์แปลงเป็น H.264 แล้ว เพื่อให้เปิดเล่นใน browser / Obsidian / GitHub ได้
- ใช้ **YOLO11n + ByteTrack** ให้รถแต่ละคันมี ID แล้วนับเมื่อจุดกึ่งกลางของกล่องข้ามเส้นนับ (เส้นแดงที่ระดับ 60% ของความสูงภาพ) แต่ละ ID นับแค่ครั้งเดียว ตั้ง `conf = 0.37` และไม่ใช้ Median filter (`DENOISE = False`) เพราะภาพจากคลิปนี้ไม่มี noise
- ผล: นับได้ **car 6 คัน** ความเร็วทั้ง pipeline (detect + track + วาดผล + เขียนไฟล์) **20.9 FPS** บน T4 ซึ่งใกล้ real-time
- 🎞️ วิดีโอผลลัพธ์: [results/tracking_counting_demo.mp4](results/tracking_counting_demo.mp4)
- ⚠️ ยังไม่ได้นับด้วยมือเพื่อเทียบความถูกต้อง ควรทำในสัปดาห์ 2 (Counting accuracy = นับได้ / นับจริง)

---

## 6. ข้อจำกัด (Limitations)

1. **ภาพกลางคืนมีน้อย (1.4%)** ความแม่นยำตอนกลางคืนจึงยังไม่ได้ทดสอบกับภาพจริงมากพอ ผลกลางคืนในรายงานนี้มาจากการจำลอง
2. **Dataset มาจากอินโดนีเซีย** สภาพถนนใกล้เคียงไทย แต่ไม่มีรถเฉพาะของไทย เช่น สองแถว ตุ๊กตุ๊ก
3. **ภาพจากกล้องตัวเดียวกันอาจอยู่ทั้งใน train และ test** ค่าบน test อาจสูงกว่าการใช้งานกับกล้องใหม่
4. **ภาพถูก resize แบบ stretch เป็น 640×640** สัดส่วนรถในข้อมูลจึงบิดไปจากภาพวิดีโอจริงเล็กน้อย
5. **Label ไม่ครบในบางภาพ** เช่น ภาพ 2 ในข้อ 5.4 ไม่ได้ label มอเตอร์ไซค์เลย ทำให้ Precision (โดยเฉพาะ motorcycle) ที่วัดได้ต่ำกว่าความจริง ถ้ามีเวลาควรตรวจและแก้ label ใน Roboflow
6. **ยังไม่ได้วัดความถูกต้องของการนับในวิดีโอ** เทียบกับการนับด้วยคน
7. **ไม่มี class จักรยาน (bicycle)** โมเดลเป็นแบบ closed-set รู้จักแค่ 4 class เมื่อเจอจักรยานซึ่งมีสองล้อและมีคนขี่เหมือนมอเตอร์ไซค์ จึงทายเป็น `motorcycle` ด้วย confidence สูง (เช่น 0.8 ในภาพ 2) ซึ่งจะทำให้ระบบ**นับมอเตอร์ไซค์เกิน**ในถนนที่มีจักรยาน

## 7. แนวทางปรับปรุง

- Train นานขึ้น (100 epoch) เพราะ best epoch คือ epoch สุดท้าย หรือลอง `yolo11s` แล้วเทียบ mAP กับความเร็ว
- เพิ่มภาพกลางคืนจริงและภาพจากกล้องในไทย (เช่น dataset Thailand-vehicles) เพื่อแก้ข้อจำกัด 1–2
- ใช้ `imgsz` ใหญ่ขึ้น (เช่น 960) หรือ SAHI (slicing inference) เพื่อช่วยตรวจวัตถุเล็กที่อยู่ไกล
- แก้ปัญหาจักรยาน: (ก) เพิ่ม class `bicycle` แล้ว label เพิ่มใน Roboflow หรือ (ข) ใน app สัปดาห์ 2 รันโมเดล COCO (`yolo11n.pt` ซึ่งมี class bicycle อยู่แล้ว) ควบคู่ ถ้ากล่อง motorcycle ทับกับกล่อง bicycle ของ COCO มาก (IoU > 0.5) ให้เปลี่ยนเป็น bicycle และไม่นับ
- ใน app สัปดาห์ 2: ใช้ `conf=0.37`, ใส่ Median filter เป็นตัวเลือกเมื่อภาพมี noise, **ปิด** Gamma/CLAHE เป็นค่าเริ่มต้น

---

## 8. Methodology (ขั้นตอนใน [Notebook บน Google Colab](https://colab.research.google.com/drive/152x_H4Ozg6T5p38-fK-j69LdWWBjTELz))

🔗 เปิด notebook: [Dataset_Model_SmartTraffic.ipynb](https://colab.research.google.com/drive/152x_H4Ozg6T5p38-fK-j69LdWWBjTELz)

```
Roboflow (YOLOv11 format)
   → Data Preparation (แก้ data.yaml, ตรวจ split, ตรวจ label)
   → Dataset Analysis (จำนวน, class, ขนาดวัตถุ, ความสว่าง)
   → Image Processing (ทดลอง → เลือก Median filter + Letterbox → src/preprocessing.py)
   → Fine-tune YOLO11n (COCO pretrained, 50 epochs)
   → Evaluation บน Test set (mAP, P, R, F1, Confusion Matrix)
   → Robustness Test (Night / Noise ± Image Processing)
   → ทดสอบ Detect + นับบนภาพ และวิดีโอ (ByteTrack + เส้นนับ)
   → best.pt + model_info.json + results/
```

| Section ใน notebook | ทำอะไร | ไฟล์ผลลัพธ์ |
|---|---|---|
| 0 Install & Config | ติดตั้ง ultralytics, roboflow, ตั้งค่า, mount Drive | — |
| 1 Download | โหลด dataset ผ่าน Roboflow API | `/content/datasets/...` |
| 2 Data Preparation | แก้ path, ตรวจ/สร้าง test split | `data_fixed.yaml` |
| 3 Dataset Analysis | สถิติและภาพตัวอย่าง | `dataset_*.csv/png/jpg` |
| 4 Image Processing | ทดลองเทคนิค, เขียน `preprocessing.py` | `letterbox_demo.png`, `image_processing_scenarios.png`, `histogram_before_after.png` (+ `image_processing_demo.png` ในภาคผนวก) |
| 5 Train | Fine-tune YOLO11n | `runs/yolo11n_vehiclecount_v5/`, `training_curves.png`, `results.png` |
| 6 Evaluation | วัดบน test set + ตาราง confusion matrix ที่ conf 0.37 (6.1) | `test_metrics_*.csv`, `confusion_table.png/csv`, `precision_recall_ap_per_class.csv`, `confusion_matrix*.png`, `Box*_curve.png` |
| 7 Robustness | วัดผลของ Image Processing | `robustness.csv/png` |
| 8 Test | Detect + นับบนภาพและวิดีโอ | `detection_*.jpg`, `tracking_counting_demo.mp4` |
| 9 Summary | สรุปผลและไฟล์ส่งต่อ | `week1_summary.md`, `best.pt`, `model_info.json` |

---

## 9. ไฟล์ที่ส่งต่อให้ทีมสัปดาห์ 2

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `best.pt` | โมเดลที่ train แล้ว (5.5 MB) |
| `model_info.json` | class names `[bus, car, motorcycle, truck]`, imgsz 640, recommended_conf 0.37 |
| `src/preprocessing.py` | `from src.preprocessing import median_filter, letterbox, ...` |
| Section 8.2 ใน notebook | โค้ดตัวอย่าง ByteTrack + เส้นนับรถ |

---

## เครดิต Dataset (ใส่ใน README)

> Vehicle Counting Dataset (v5) by Pram, Roboflow Universe, https://universe.roboflow.com/pram/vehicle-counting-flusm — licensed under CC BY 4.0


---

## ภาคผนวก ก: การสำรวจเทคนิค Image Processing เบื้องต้น

ก่อนเลือกเทคนิค ได้ลองดูผลของเทคนิคพื้นฐานอื่นด้วย (Grayscale, Gaussian Blur, Histogram Equalization, Canny Edge) แต่ไม่ได้นำไปใช้ใน pipeline เพราะ YOLO ต้องการภาพสี 3 ช่อง และไม่มีปัญหาที่เทคนิคเหล่านี้แก้ได้ตรงจุดในงานนี้

![รูปที่ ก.1 ผลของเทคนิค Image Processing พื้นฐาน 8 แบบ (สำรวจเบื้องต้น)](results/image_processing_demo.png)
*รูปที่ ก.1 ผลของเทคนิค Image Processing พื้นฐาน 8 แบบ (สำรวจเบื้องต้น)*
