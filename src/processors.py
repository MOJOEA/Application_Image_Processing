import cv2
import numpy as np
import tempfile
from src.preprocessing import preprocess_frame
from src.inference import predict_objects

### ฟังก์ชัน: ประมวลผลภาพส่งคืนผลดังนี้ ภาพต้นฉบับ, ภาพจากที่ประมวลได้, ข้อมูลรูปแบบ JSON
def process_image_file(uploaded_file, model, enable_denoise, img_size, conf_thresh):
    """จัดการไฟล์ประเภทรูปภาพ (เหมือนเดิม)"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    processed_bgr = preprocess_frame(img_bgr, denoise=enable_denoise)
    res_plotted, detected_objects = predict_objects(model, processed_bgr, img_size, conf_thresh)
    
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return orig_rgb, res_rgb, detected_objects

### ฟังก์ชัน: ประมวลผลวิดิโอส่งคืนผลดังนี้ เหรมวิดิโอที่ประมวลได้, ข้อมูลรูปแบบ JSON
def process_video_file(uploaded_file, model, enable_denoise, img_size, conf_thresh, 
                       frame_placeholder, data_placeholder, speed_multiplier):
    """จัดการไฟล์ประเภทวิดีโอ (เปลี่ยนเป็นระบบ Frame Skipping เพื่อเร่งความเร็ว)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    skip_interval = max(1, int(speed_multiplier)) 
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1   
        if frame_count % skip_interval != 0:
            continue
        
        processed_frame = preprocess_frame(frame, denoise=enable_denoise)
        frame_plotted, detected_objects = predict_objects(model, processed_frame, img_size, conf_thresh)
        
        frame_rgb = cv2.cvtColor(frame_plotted, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=False)
        data_placeholder.json(detected_objects)
        
    cap.release()
