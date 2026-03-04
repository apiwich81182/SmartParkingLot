import streamlit as st
import cv2
import numpy as np
import pickle
import time

# ==========================================
# 1. การตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(page_title="CV Pipeline Dashboard", layout="wide")
st.title("🚗 Interactive Parking Analysis & CV Pipeline")

# ==========================================
# 2. Sidebar: ปรับค่า Parameter สำหรับ Pipeline
# ==========================================
st.sidebar.header("🛠️ Global Controls")
playback_delay = st.sidebar.slider("Delay (sec)", 0.0, 0.5, 0.0)

st.sidebar.markdown("---")

# --- การตั้งค่าสำหรับ Camera 1 ---
with st.sidebar.expander("🎥 Camera 1 Settings", expanded=True):
    m_blur_1 = st.slider("Median Blur (Cam 1)", 1, 15, 3, step=2, key="mb1")
    g_blur_1 = st.slider("Gaussian Blur", 1, 15, 5, step=2, key="gb1")
    dilate_1 = st.slider("Dilation Iterations", 1, 5, 2, key="di1")
    c_low_1 = st.slider("Canny Low (Cam 1)", 10, 150, 10, key="cl1")
    c_high_1 = st.slider("Canny High (Cam 1)", 100, 300, 130, key="ch1")
    cv_limit_1 = st.slider("Decision Limit (Cam 1)", 100, 1000, 730, key="lim1")

# --- การตั้งค่าสำหรับ Camera 2 ---
with st.sidebar.expander("🎥 Camera 2 Settings", expanded=True):
    m_blur_2 = st.slider("Median Blur (Cam 2)", 1, 15, 5, step=2, key="mb2")
    g_blur_2 = st.slider("Gaussian Blur", 1, 15, 5, step=2, key="gb2")
    dilate_2 = st.slider("Dilation Iterations", 1, 5, 3, key="di2")
    c_low_2 = st.slider("Canny Low (Cam 2)", 10, 150, 30, key="cl2")
    c_high_2 = st.slider("Canny High (Cam 2)", 100, 300, 170, key="ch2")
    cv_limit_2 = st.slider("Decision Limit (Cam 2)", 10, 300, 160, key="lim2")

# ==========================================
# 3. ฟังก์ชันประมวลผล (Pipeline)
# ==========================================
# เพิ่มพารามิเตอร์ g_blur และ d_iter เข้าไป
def run_pipeline(img, m_blur, g_blur, c_low, c_high, d_iter):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, m_blur)
    
    # 🌟 ใช้ค่าจาก Slider g_blur
    blur = cv2.GaussianBlur(median, (g_blur, g_blur), 1)
    
    canny = cv2.Canny(blur, c_low, c_high)
    kernel = np.ones((3, 3), np.uint8)
    
    # 🌟 ใช้ค่าจาก Slider d_iter
    dilate = cv2.dilate(canny, kernel, iterations=d_iter)
    
    return {
        "Gray Scale": gray,
        "Median Blur": median,
        "Gaussian Blur": blur,
        "Canny Edge": canny,
        "Dilation": dilate
    }

def process_parking(img, posList, threshold, dilate_img):
    spaceCounter = 0
    for pos in posList:
        px1, py1, px2, py2 = map(int, pos)
        pad_w, pad_h = int((px2-px1)*0.15), int((py2-py1)*0.15)
        
        # ใช้ภาพจากขั้นตอน Dilation มานับพิกเซล
        imgCrop = dilate_img[py1+pad_h:py2-pad_h, px1+pad_w:px2+pad_w]
        count = cv2.countNonZero(imgCrop)

        color, thick = ((0, 255, 0), 2) if count < threshold else ((0, 0, 255), 1)
        if count < threshold: spaceCounter += 1
        
        cv2.rectangle(img, (px1, py1), (px2, py2), color, thick)
        cv2.putText(img, str(count), (px1+3, py1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
    return img, spaceCounter

# ==========================================
# 4. เตรียมไฟล์และ UI Tabs
# ==========================================
configs = [
    {"video": "park1.mp4", "pickle": "ParkingPos1.pickle", "thresh": cv_limit_1},
    {"video": "park2.mp4", "pickle": "ParkingPos2.pickle", "thresh": cv_limit_2}
]

caps = [cv2.VideoCapture(c["video"]) for c in configs]
posLists = []
for c in configs:
    try:
        with open(c["pickle"], 'rb') as f: posLists.append(pickle.load(f))
    except: posLists.append([])

# --- แก้ไขส่วนเตรียม UI Tabs ---
tab1, tab2 = st.tabs(["📺 Live Dashboard", "🔍 Processing Pipeline"])

with tab1:
    col1, col2 = st.columns(2)
    p_holders = [col1.empty(), col2.empty()]
    s_texts = [col1.empty(), col2.empty()]

# --- 1. แก้ไขส่วนเตรียม UI ใน Tab 2 ---
with tab2:
    st.subheader("🛠️ Step-by-Step Visualization")
    
    # จองพื้นที่สำหรับ Camera 1
    st.caption("🎥 Camera 1 Pipeline")
    row1_cols = st.columns(5)
    step_holders_1 = [col.empty() for col in row1_cols]
    
    st.markdown("---") # เส้นคั่นระหว่างกล้อง
    
    # จองพื้นที่สำหรับ Camera 2
    st.caption("🎥 Camera 2 Pipeline")
    row2_cols = st.columns(5)
    step_holders_2 = [col.empty() for col in row2_cols]

# ==========================================
# 5. Main Loop
# ==========================================
while True:
    # เก็บค่าพารามิเตอร์ใส่ลิสต์เพื่อให้ดึงใช้ง่ายตามรอบของ Loop (i=0, i=1)
    m_blurs = [m_blur_1, m_blur_2]
    g_blurs = [g_blur_1, g_blur_2]  # 🌟 เพิ่มบรรทัดนี้
    c_lows = [c_low_1, c_low_2]
    c_highs = [c_high_1, c_high_2]
    d_iters = [dilate_1, dilate_2]  # 🌟 เพิ่มบรรทัดนี้
    limits = [cv_limit_1, cv_limit_2]

    for i in range(2):
        success, img = caps[i].read()
        if not success:
            caps[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # 🌟 รัน Pipeline โดยใช้ค่าที่แยกกันตาม i (0 หรือ 1)
        # 🌟 แก้ไขบรรทัดนี้ให้ส่งค่าครบตามที่ฟังก์ชันต้องการ
        steps = run_pipeline(img, m_blurs[i], g_blurs[i], c_lows[i], c_highs[i], d_iters[i])
        
        # ประมวลผลที่จอดรถโดยใช้ Threshold และภาพ Dilation ของกล้องนั้นๆ
        processed_img, count = process_parking(img.copy(), posLists[i], limits[i], steps["Dilation"])
        
        # แสดงผล Tab 1 (Dashboard)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        p_holders[i].image(img_rgb, use_container_width=True)
        s_texts[i].markdown(f"### 🟢 Camera {i+1} Free: {count} / {len(posLists[i])}")

        # แสดงผล Tab 2 (Pipeline): แยกกล้อง 1 ไว้แถวบน กล้อง 2 ไว้แถวล่าง
        if i == 0:
            for idx, (name, step_img) in enumerate(steps.items()):
                step_holders_1[idx].image(step_img, caption=f"Cam1: {name}", use_container_width=True)
        else:
            for idx, (name, step_img) in enumerate(steps.items()):
                step_holders_2[idx].image(step_img, caption=f"Cam2: {name}", use_container_width=True)

    if playback_delay > 0:
        time.sleep(playback_delay)