

โครงสร้างสตัคเจอรืงาน

# Application Image Processing

เว็บแอปพลิเคชันสำหรับประมวลผลรูปภาพและตรวจจับวัตถุด้วย AI พัฒนาขึ้นโดยใช้ **Streamlit** และโมเดล **YOLO (PyTorch)** รองรับการทำงานแบบ Multi-page ทั้งหน้าตรวจจับวัตถุและหน้าแสดงผลสถิติ (Dashboard)

---

## โครงสร้างโปรเจกต์ (Project Structure)

```text
├── .streamlit/          # ไฟล์ตั้งค่าการแสดงผลของ Streamlit (config.toml)
├── .venv/               # สภาพแวดล้อมจำลอง (Virtual Environment)
├── data/                # โฟลเดอร์เก็บข้อมูลดิบและรูปภาพสำหรับทดสอบ
├── doc/                 # เอกสารประกอบโครงการ
│
├── models/              # โฟลเดอร์เก็บโมเดล AI
│   ├── best.pt          # ไฟล์ Weight ของโมเดล YOLO
│   └── model_info.json  # รายละเอียดและ Class ของโมเดล
│
├── src/                 # ซอร์สโค้ดหลักในการประมวลผล (Core Logic)
│   ├── inference.py     # โค้ดสำหรับรันโมเดลทำนายผล (AI Inference)
│   ├── preprocessing.py # โค้ดสำหรับการเตรียมรูปภาพ (Image Preprocessing)
│   └── processors.py    # โค้ดประมวลผลข้อมูลและภาพหลังจากการตรวจจับ
│
├── static/              # ไฟล์คงที่ เช่น รูปภาพโลโก้ หรือ ไอคอน
├── styles/              # ไฟล์สำหรับตกแต่งหน้าเว็บ (CSS)
│
├── views/               # หน้าอินเตอร์เฟสย่อยของแอปพลิเคชัน
│   ├── dashboard_page.py# หน้าแสดงสถิติและกราฟสรุปผล
│   └── detection_page.py# หน้าอัปโหลดภาพและตรวจจับวัตถุ
│
├── app.py               # ไฟล์หลักสำหรับรันแอปพลิเคชัน (Main Entry Point)
└── requirements.txt     # รายการ Library ที่โปรเจกต์จำเป็นต้องใช้
```

---

## เครื่องมือที่ใช้พัฒนา (Tech Stack)

* **Backend & Frontend:** [Python](https://python.org) (เวอร์ชัน 3.x), [Streamlit](https://streamlit.io)
* **AI & Computer Vision:** [PyTorch](https://pytorch.org) (YOLOv8/v9), OpenCV
* **Styling:** Custom CSS

---

## ขั้นตอนการติดตั้งและรันโปรเจกต์

### 1. ดาวน์โหลดโปรเจกต์ (Clone Repository)
```bash
git clone https://github.com[ชื่อผู้ใช้ของคุณ]/Application-Image-Processing.git
cd Application-Image-Processing
```

### 2. สร้างและเปิดใช้งาน Virtual Environment
* **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. ติดตั้ง Library ที่จำเป็น
```bash
pip install -r requirements.txt
```

### 4. สั่งรันแอปพลิเคชัน
```bash
python -m streamlit run app.py
```
หลังจากรันคำสั่ง ระบบจะเปิดเว็บเบราว์เซอร์ขึ้นมาที่หน้า `http://localhost:8501` โดยอัตโนมัติ

---

## ฟีเจอร์การใช้งาน (Features)

* **Image Object Detection:** อัปโหลดรูปภาพเพื่อตรวจจับวัตถุผ่านโมเดล AI (`best.pt`) พร้อมแสดงกรอบ Bounding Box แบบ Real-time
* **Video Object Detection:** อัปโหลดวิดิโอเพื่อตรวจจับวัตถุผ่านโมเดล AI (`best.pt`) พร้อมแสดงกรอบ Bounding Box แบบ Real-time
* **Advanced Preprocessing:** ระบบปรับแต่งและเตรียมไฟล์เบื้องหลังก่อนส่งให้ AI เพื่อเพิ่มความแม่นยำสูงสุด โดยรองรับฟังก์ชันดังนี้:
  * **Noise Reduction:** ตัดสัญญาณรบกวนออกจากรูปภาพหรือวิดีโอ
  * **Resolution Scaling:** ปรับขนาดของไฟล์ฝั่งรับเข้า (Input) และสัดส่วนการแสดงผลผลลัพธ์ (Output)
  * **Frame Rate Control:** ปรับความเร็วในการอ่านเฟรมของวิดีโอได้ตั้งแต่ความเร็วปกติจนถึงสองเท่า (**1x - 2x**)* **Interactive Dashboard:** "Demo" หน้าต่างสรุปผลข้อมูล สถิติ และจำนวนวัตถุที่ตรวจจับได้ในรูปแบบกราฟที่สวยงาม
* **Multi-page Interface:** แยกสัดส่วนหน้าการทำงานอย่างชัดเจนผ่านระบบหน้าต่างย่อยของ Streamlit

---

## ผู้จัดทำโครงการ (Contributors)

### ผู้จัดทำระบบและโครงสร้างงาน (System & Workflow Architect)
[![GitHub Profile](https://vercel.app)](https://github.com/MOJOEA)
* **ชื่อผู้พัฒนา:** Punpun (MOJOEA)
* **บทบาท:** ออกแบบ System Architecture, พัฒนาหน้าอินเตอร์เฟสระบบ Multi-page ด้วย Streamlit และวางระบบ Image/Video Preprocessing

---

### ทีมพัฒนาโมเดล AI (AI Model Developer)
[![GitHub Profile]()

* **ชื่อผู้พัฒนา:** [ชื่อเพื่อน]
* **บทบาท:** พัฒนาและฝึกสอนโมเดลปัญญาประดิษฐ์ (AI Model Training), จัดทำไฟล์ Weight โมเดล (`best.pt`) และโครงสร้างข้อมูลคลาสวัตถุ (`model_info.json`)

