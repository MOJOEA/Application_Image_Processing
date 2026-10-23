'''Image preprocessing for Smart Traffic Monitoring (input/output = BGR uint8 จาก OpenCV)

ใช้จริงใน pipeline : median_filter, letterbox, preprocess_frame
ทดลองแล้วไม่ใช้     : gamma_correction, auto_gamma, clahe, enhance_low_light (ทำให้ภาพกลางคืนแย่ลง −0.168 mAP50)
ใช้สร้างชุดทดสอบ    : simulate_low_light, add_salt_pepper
'''
import cv2
import numpy as np


# ======================= ใช้จริง =======================
def median_filter(img, k=3):
    '''ลด noise แบบจุด (salt & pepper): แทนพิกเซลด้วยค่ากลางของหน้าต่าง k×k — ไม่ทำให้ขอบรถเบลอ'''
    return cv2.medianBlur(img, k)


def letterbox(img, size=640, color=(114, 114, 114)):
    '''Resize โดยรักษาสัดส่วน (r = size / max(h, w)) แล้วเติมขอบสีเทาให้เป็น size×size (แบบเดียวกับ YOLO)'''
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def preprocess_frame(img, denoise=False):
    '''ขั้นตอนก่อนส่งเข้า YOLO ใน app (letterbox ทำภายใน YOLO อัตโนมัติอยู่แล้วตอน predict/track)'''
    return median_filter(img, 3) if denoise else img


# ================== ทดลองแล้วไม่ใช้ ==================
def gamma_correction(img, gamma=1.5):
    '''out = 255 * (in/255)^(1/gamma) ; gamma > 1 = สว่างขึ้น'''
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def auto_gamma(img, target=0.45):
    '''เลือก gamma อัตโนมัติให้ความสว่างเฉลี่ยเข้าใกล้ target'''
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    mean = min(max(mean, 1e-3), 0.999)
    gamma = np.log(mean) / np.log(target)
    return gamma_correction(img, float(np.clip(gamma, 0.5, 4.0)))


def clahe(img, clip=2.0, tile=8):
    '''CLAHE บนช่อง L ของ LAB (สีไม่เพี้ยน)'''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def brightness(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean())


def enhance_low_light(img, dark_thresh=80):
    '''auto gamma → CLAHE เฉพาะภาพมืด (ใช้ในการทดลองเท่านั้น ไม่ใช้ใน pipeline จริง)'''
    if brightness(img) < dark_thresh:
        return clahe(auto_gamma(img), clip=2.0)
    return img


# ============ ใช้สร้างสถานการณ์ทดสอบ (robustness test) ============
def simulate_low_light(img, factor=0.35, gamma=1.6, noise_sigma=6, seed=0):
    rng = np.random.default_rng(seed)
    dark = ((img / 255.0) ** gamma) * factor * 255
    dark = dark + rng.normal(0, noise_sigma, img.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.04, seed=0):
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out
