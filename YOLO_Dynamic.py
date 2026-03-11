import cv2
import pickle
import numpy as np
from ultralytics import YOLO

# 1. โหลดโมเดล YOLO
model = YOLO('runs/detect/train7/weights/best.pt')

# 2. โหลดพิกัดช่องจอด
with open('ParkingPos3.pickle', 'rb') as f:
    posList = pickle.load(f)

# ส่วนที่ 1: เตรียมระบบ Feature Matching (ORB)
ref_img = cv2.imread('parking_image3.jpg', cv2.IMREAD_GRAYSCALE)

# สร้างตัวค้นหาจุดเด่น ORB
orb = cv2.ORB_create(1000)
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

# สร้างตัวจับคู่ (Matcher)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 3. เปิดวิดีโอ
cap = cv2.VideoCapture('park3.mp4')

# ตั้งค่าบันทึกวิดีโอ
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('smart_parking_dynamic3.mp4', fourcc, fps, (frame_width, frame_height))

print("Starting Dynamic Parking Detection... PRESS 'q' TO STOP")

while True:
    success, img = cap.read()
    if not success:
        print("End of video. Saving file...")
        break

    # ส่วนที่ 2: หาค่าการสั่นของกล้อง (Homography)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(img_gray, None)
    
    H = None # ตัวแปรเก็บเมทริกซ์การเคลื่อนที่
    
    if des_curr is not None and len(des_curr) > 0:
        # จับคู่จุดเด่นระหว่างภาพต้นฉบับ กับ เฟรมปัจจุบัน
        matches = matcher.match(des_ref, des_curr)
        # เรียงลำดับเอาเฉพาะจุดที่เหมือนกันที่สุด
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) >= 4:
            # ดึงพิกัด (x, y) ของจุดที่จับคู่ได้
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # คำนวณหา Homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 4. ให้ Model ตรวจจับรถ
    results = model.predict(img, classes=[3, 4, 5, 8], conf=0.15, verbose=False)
    car_boxes = results[0].boxes.xyxy.cpu().numpy()

    spaceCounter = 0

    # 5. เช็คว่ามีรถอยู่ในช่องจอดไหม
    for pos in posList:
        px1, py1, px2, py2 = map(int, pos)
        
        # ส่วนที่ 3: ขยับกรอบตามกล้อง
        # สร้างพิกัด 4 มุมของกรอบเดิม (บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย)
        pts = np.float32([[px1, py1], [px2, py1], [px2, py2], [px1, py2]]).reshape(-1, 1, 2)
        
        if H is not None:
            # เอาเมทริกซ์ H มาคูณเพื่อหาตำแหน่งกรอบใหม่ในเฟรมนี้
            transformed_pts = cv2.perspectiveTransform(pts, H)
        else:
            transformed_pts = pts # ถ้าภาพเบลอจัดจนหา H ไม่ได้ ให้อยู่ที่เดิมไปก่อน
            
        # ปัดเศษเป็นจำนวนเต็มเตรียมไว้วาดรูป
        poly_pts = np.int32(transformed_pts)

        is_occupied = False
        
        for car in car_boxes:
            cx1, cy1, cx2, cy2 = map(int, car)
            car_center_x = int((cx1 + cx2) / 2)
            car_center_y = int((cy1 + cy2) / 2)
            
            # ส่วนที่ 4: เช็คว่ารถอยู่ในพื้นที่รูปหลายเหลี่ยมไหม
            # ใช้ cv2.pointPolygonTest เช็คว่าจุดกึ่งกลางรถ อยู่ในกรอบที่เอียงแล้วหรือเปล่า
            if cv2.pointPolygonTest(np.float32(poly_pts), (car_center_x, car_center_y), False) >= 0:
                is_occupied = True
                break
                
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        thickness = 2
        if not is_occupied: spaceCounter += 1
            
        # ใช้ polylines วาดกรอบรูปหลายเหลี่ยม
        cv2.polylines(img, [poly_pts], True, color, thickness)
        
    cv2.rectangle(img, (0, 0), (250, 60), (0, 0, 0), -1)
    cv2.putText(img, f"Free: {spaceCounter}/{len(posList)}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dynamic Parking Detection", img)
    out.write(img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved successfully")