import streamlit as st
from src.inference import load_model
from src.processors import process_image_file, process_video_file

### ฟังก์ชัน: โหรต Styles css
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

### ฟังก์ชัน: วาดUI แสดงผล
def render():
    st.set_page_config(layout="wide")
    load_css("styles/detection_page.css")
    
    st.sidebar.markdown("<h2 style='color: #58a6ff;'>ตั้งค่าระบบ</h2>", unsafe_allow_html=True)
    
    ### สวิชควบคุม: เปิดใช้งานระบบลดสัญญาณรบกวน
    enable_denoise = st.sidebar.checkbox("เปิดใช้งานระบบลดสัญญาณรบกวน (Denoise)", value=True)
    st.sidebar.markdown("---")
    
    ### แถบควบคุม: ขนาดของภาพในการแสดงผล
    display_width = st.sidebar.slider("ปรับขนาดแสดงผล (พิกเซล)", min_value=200, max_value=1200, value=500, step=50)
    ### แถบควบคุม: ความเร็วในการอ่านและแสดงเฟรมวิดิโอ
    video_speed = st.sidebar.select_slider("ความเร็วในการเล่นวิดีโอ",
        options=[1, 1.25, 1.5, 1.75, 2],
        value=1,
        format_func=lambda x: "ปกติ (1x)" if x == 1 else f"เร็วขึ้น {x}x"
    )

    st.markdown(
        f"<style>.stVideo, div[data-testid='stImage'] img {{ width: {display_width}px !important; }}</style>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-title">ระบบตรวจจับวัตถุอัจฉริยะ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">รองรับการตรวจสอบทั้ง ภาพเดี่ยว, หลายภาพ และ วิดีโอ</div>', unsafe_allow_html=True)


    CONF_THRESHOLD = 0.37
    IMAGE_SIZE = 600

    model = load_model("models/best.pt")

    uploaded_files = st.file_uploader(
        "เลือกไฟล์ของคุณ (รองรับ .jpg, .jpeg, .png, .mp4)", 
        type=["jpg", "jpeg", "png", "mp4"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1].lower()
            
            st.markdown("<hr style='border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #c9d1d9;'>กำลังประมวลผลไฟล์: <span style='color: #58a6ff;'>{file_name}</span></h3>", unsafe_allow_html=True)

            if file_extension in ["jpg", "jpeg", "png"]:
                with st.spinner("AI กำลังวิเคราะห์รูปภาพ..."):
                    orig_rgb, res_rgb, detected_objects = process_image_file(
                        uploaded_file, model, enable_denoise, IMAGE_SIZE, CONF_THRESHOLD
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(orig_rgb, caption="รูปภาพต้นฉบับ", use_container_width=True)
                with col2:
                    st.image(res_rgb, caption="ผลลัพธ์การตรวจจับ", use_container_width=True)
                    
                with st.expander("ดูข้อมูลพิกัดและค่าความมั่นใจ (JSON)"):
                    st.json(detected_objects)

            elif file_extension == "mp4":
                frame_placeholder = st.empty()
                data_placeholder = st.empty()
                
                process_video_file(
                    uploaded_file, model, enable_denoise, IMAGE_SIZE, CONF_THRESHOLD, 
                    frame_placeholder, data_placeholder, speed_multiplier=video_speed
                )
    else:
        st.info("พร้อมใช้งาน: คุณสามารถลากไฟล์มาวางในช่องด้านบนได้ทันที")

if __name__ == "__main__":
    render()
