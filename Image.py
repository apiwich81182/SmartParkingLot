import cv2
import numpy as np
import pickle
import os

# 1. ตั้งค่าพารามิเตอร์
p = {
    'm_blur': 3, 'g_blur': 5, 'c_low': 30, 'c_high': 170, 'd_iter': 2, 'limit': 120, 
    'pickle': 'ParkingPos3.pickle', 'video': 'park3.mp4'
}

# 2. โหลดพิกัดช่องจอด
if not os.path.exists(p['pickle']) or not os.path.exists(p['video']):
    print("Error: Pickle file or Video file not found!")
    exit()
    
with open(p['pickle'], 'rb') as f:
    posList = pickle.load(f)

# 3. เปิดวิดีโอและเตรียม VideoWriter
cap = cv2.VideoCapture(p['video'])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('smart_parking_static_dip.mp4', fourcc, fps, (width, height))

print("Processing with Static ROI... PRESS 'q' TO STOP")

while True:
    success, img = cap.read()
    if not success:
        print("Video ended.")
        break
        
    # 4. Image Processing Pipeline 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, p['m_blur'])
    blur = cv2.GaussianBlur(median, (p['g_blur'], p['g_blur']), 1)
    canny = cv2.Canny(blur, p['c_low'], p['c_high'])
    dilate = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=p['d_iter'])
    
    spaceCounter = 0
    
    # 5. ตรวจสอบพิกเซลในกรอบ
    for pos in posList:
        px1, py1, px2, py2 = map(int, pos)
        
        # คำนวณขอบเขตโดยตัดขอบออก 15% (Padding) 
        pad_w, pad_h = int((px2 - px1) * 0.15), int((py2 - py1) * 0.15)
        
        # ตัดภาพ
        y_start = max(0, py1 + pad_h)
        y_end = min(dilate.shape[0], py2 - pad_h)
        x_start = max(0, px1 + pad_w)
        x_end = min(dilate.shape[1], px2 - pad_w)
        
        if y_end <= y_start or x_end <= x_start:
            continue

        imgCrop = dilate[y_start:y_end, x_start:x_end]
        
        # นับจุดพิกเซลสีขาว
        count = cv2.countNonZero(imgCrop)
        
        # กำหนดสีและวาดกรอบ
        if count < p['limit']:
            color, thick = (0, 255, 0), 2 
            spaceCounter += 1
        else:
            color, thick = (0, 0, 255), 1 
        
        # วาดสี่เหลี่ยมผืนผ้า
        cv2.rectangle(img, (px1, py1), (px2, py2), color, thick)
        cv2.putText(img, str(count), (px1 + 3, py1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # วาดพื้นหลังและแสดงผลรวมจำนวนช่องว่าง
    cv2.rectangle(img, (0, 0), (250, 60), (0, 0, 0), -1)
    cv2.putText(img, f"Free: {spaceCounter}/{len(posList)}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงผลบนหน้าจอและบันทึกลงไฟล์
    cv2.imshow("Static Image Processing", img)
    out.write(img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# คืนทรัพยากรระบบ
cap.release()
out.release()
cv2.destroyAllWindows()
print("Success! Saved to 'smart_parking_static_dip.mp4'")
