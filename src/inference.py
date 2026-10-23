import streamlit as st
from ultralytics import YOLO

@st.cache_resource

# ======================= ใช้จริง =======================

### ฟังก์ชัน: เลือกโหรตโมเดล
def load_model(model_path):
    """โหลดโมเดลและเก็บไว้ใน Cache"""
    return YOLO(model_path)

### ฟังก์ชัน: ส่งให้โมเดลประมวลผลส่งคืนผลดังนี้ ภาพหรือเฟรมจากที่ประมวลได้, ข้อมูลรูปแบบ JSON
def predict_objects(model, source_img, img_size=640, conf_thresh=0.37):
    """ส่งภาพให้โมเดลทำ Inference และแปลงผลลัพธ์เป็น JSON format"""
    results = model.predict(source=source_img, imgsz=img_size, conf=conf_thresh, verbose=False)
    
    detected_objects = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        detected_objects.append({
            "class": results[0].names[class_id],
            "confidence": round(float(box.conf), 3),
            "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()]
        })
    return results[0].plot(), detected_objects

# ================== ทดลองแล้วไม่ใช้ ==================
# ============= ใช้สร้างสถานการณ์ทดสอบ ================
